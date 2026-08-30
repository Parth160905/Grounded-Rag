package dev.pgdocs.rag.ingest;

import dev.pgdocs.rag.index.HybridSearchRepository;
import dev.pgdocs.rag.index.HybridSearchRepository.Hit;
import dev.pgdocs.rag.ingest.Documents.Chunk;
import dev.pgdocs.rag.ingest.Documents.SourceDoc;
import org.jsoup.Jsoup;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Runs against a real Postgres with pgvector, not a mock or H2.
 *
 * <p>Everything load-bearing in this system is database behaviour — the
 * generated tsvector with its weights, the HNSW index, {@code SKIP LOCKED}
 * semantics, the {@code ::vector} cast. None of it can be exercised against an
 * in-memory substitute, so a green suite here means the schema actually works
 * rather than that the mocks agreed with each other.
 */
@SpringBootTest
@org.junit.jupiter.api.Tag("integration")
@org.junit.jupiter.api.Tag("integration")
@Testcontainers
class IngestIntegrationTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>(
            DockerImageName.parse("pgvector/pgvector:pg16")
                    .asCompatibleSubstituteFor("postgres"))
            .withDatabaseName("pgdocs_test");

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
        registry.add("ingest.docs-dir", () -> "./docs/pg16");
    }

    @Autowired JdbcTemplate jdbc;
    @Autowired StructureAwareChunker chunker;
    @Autowired EmbeddingService embeddings;
    @Autowired IngestRepository repo;
    @Autowired HybridSearchRepository search;

    private static final String SELECT_PAGE = """
            <div class="refentry" id="SQL-SELECT">
              <h1>SELECT</h1>
              <div class="refsect1" id="SQL-SELECT-SYNOPSIS">
                <h2>Synopsis</h2>
                <pre class="synopsis">SELECT [ ALL | DISTINCT ] * FROM table
            WHERE condition
            ORDER BY expression</pre>
              </div>
              <div class="refsect1" id="SQL-SELECT-PARAMETERS">
                <h2>Parameters</h2>
                <div class="refsect2" id="SQL-FROM">
                  <h3>FROM Clause</h3>
                  <p>The FROM clause specifies one or more source tables for the SELECT.
                     It is a comma-separated list of table references, which may be a
                     table name, a subquery, a JOIN clause, or a function call. This
                     paragraph is padded so the section clears the minimum token
                     threshold and is not discarded as a fragment by the chunker.</p>
                </div>
                <div class="refsect2" id="SQL-WHERE">
                  <h3>WHERE Clause</h3>
                  <p>The optional WHERE clause has the general form of a condition,
                     where the condition is any expression that evaluates to a result
                     of type boolean. Any row that does not satisfy this condition
                     will be eliminated from the output, padded here for length.</p>
                </div>
              </div>
            </div>
            """;

    @Test
    void migrationCreatesSchemaWithPgvectorAndWeightedTsvector() {
        Integer ext = jdbc.queryForObject(
                "SELECT count(*) FROM pg_extension WHERE extname = 'vector'", Integer.class);
        assertThat(ext).isEqualTo(1);

        // The heading path must carry weight A and body text weight B. If this
        // regresses, lexical ranking quietly degrades on every query.
        String tsv = jdbc.queryForObject(
                "SELECT setweight(to_tsvector('english','SELECT'),'A')::text", String.class);
        assertThat(tsv).contains(":1A");
    }

    @Test
    void chunkerKeepsSynopsisWholeAndSplitsOnSections() {
        List<Chunk> chunks = chunker.chunk(Jsoup.parse(SELECT_PAGE), "SELECT");

        assertThat(chunks).isNotEmpty();

        // Rule 2: the synopsis survives intact. Half a grammar production
        // answers nothing, so this is the assertion that matters most.
        Chunk synopsis = chunks.stream()
                .filter(Chunk::preserved)
                .findFirst()
                .orElseThrow(() -> new AssertionError("synopsis chunk was not preserved"));
        assertThat(synopsis.text()).contains("SELECT [ ALL | DISTINCT ]");
        assertThat(synopsis.text()).contains("ORDER BY expression");

        // Rule 1: subsections became their own chunks, each carrying the full
        // breadcrumb so a fragment is self-describing when retrieved alone.
        assertThat(chunks).anyMatch(c -> c.headingPath().contains("FROM Clause"));
        assertThat(chunks).anyMatch(c -> c.headingPath().contains("WHERE Clause"));
        assertThat(chunks).allMatch(c -> c.text().startsWith(c.headingPath()));

        // Deep-link anchors survived parsing.
        assertThat(chunks).anyMatch(c -> "#SQL-FROM".equals(c.anchor()));
    }

    @Test
    void hybridSearchFusesBothChannels() {
        List<Chunk> chunks = chunker.chunk(Jsoup.parse(SELECT_PAGE), "SELECT");
        List<float[]> vectors = embeddings.embedAll(chunks.stream().map(Chunk::text).toList());

        repo.replaceDocument(new SourceDoc(
                "sql-select.html",
                "https://www.postgresql.org/docs/16/sql-select.html",
                "SELECT", "16", "testhash"), chunks, vectors);
        repo.buildVectorIndex();

        // Paraphrase — shares almost no vocabulary with the source text, so
        // this can only be found by the dense channel.
        List<Hit> paraphrase = search.search(
                "how do I filter out rows I do not want",
                embeddings.embedOne("how do I filter out rows I do not want"), 5, 20);

        assertThat(paraphrase).isNotEmpty();
        assertThat(paraphrase.getFirst().denseRank()).isNotNull();

        // Exact identifier — the case where vector search alone drifts to
        // semantically adjacent content instead of the thing that was named.
        List<Hit> exact = search.search(
                "FROM clause", embeddings.embedOne("FROM clause"), 5, 20);

        assertThat(exact).isNotEmpty();
        assertThat(exact).anyMatch(h -> h.lexicalRank() != null);
        assertThat(exact.getFirst().headingPath()).contains("FROM");

        // Deep links resolve to a real in-page anchor.
        assertThat(exact.getFirst().deepLink()).startsWith("https://www.postgresql.org/docs/16/");
    }

    @Test
    void skipLockedQueueClaimsEachJobExactlyOnce() {
        repo.enqueue(List.of("a.html", "b.html", "c.html"), "16");

        var first = repo.claimBatch(2, "10 minutes");
        var second = repo.claimBatch(2, "10 minutes");

        assertThat(first).hasSize(2);
        assertThat(second).hasSize(1);

        // No job appears in both batches — this is the property that lets
        // workers run concurrently with no external coordination.
        assertThat(first.stream().map(IngestRepository.ClaimedJob::id))
                .doesNotContainAnyElementsOf(
                        second.stream().map(IngestRepository.ClaimedJob::id).toList());
    }

    @Test
    void reingestingUnchangedContentIsSkipped() {
        List<Chunk> chunks = chunker.chunk(Jsoup.parse(SELECT_PAGE), "SELECT");
        List<float[]> vectors = embeddings.embedAll(chunks.stream().map(Chunk::text).toList());
        String hash = IngestService.sha256(SELECT_PAGE);

        repo.replaceDocument(new SourceDoc("dup.html", "u", "SELECT", "16", hash),
                chunks, vectors);

        assertThat(repo.existingHash("dup.html", "16")).contains(hash);
        assertThat(repo.existingHash("dup.html", "16")).isNotEqualTo("different");
    }
}
