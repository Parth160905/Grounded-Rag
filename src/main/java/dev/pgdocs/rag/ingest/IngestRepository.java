package dev.pgdocs.rag.ingest;

import dev.pgdocs.rag.ingest.Documents.Chunk;
import dev.pgdocs.rag.ingest.Documents.SourceDoc;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.sql.Types;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.StringJoiner;

@Repository
public class IngestRepository {

    private final JdbcTemplate jdbc;

    public IngestRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /**
     * Content-hash check so re-runs are cheap.
     *
     * <p>Ingesting 2,700 pages takes minutes; you will re-run it constantly
     * while tuning the chunker. Skipping unchanged documents turns that into
     * seconds and makes iteration bearable.
     */
    public Optional<String> existingHash(String sourcePath, String docVersion) {
        List<String> rows = jdbc.queryForList(
                "SELECT content_hash FROM document WHERE source_path = ? AND doc_version = ?",
                String.class, sourcePath, docVersion);
        return rows.isEmpty() ? Optional.empty() : Optional.of(rows.getFirst());
    }

    /**
     * Replace a document and all its chunks atomically.
     *
     * <p>Delete-then-insert rather than diffing: chunk boundaries shift when the
     * chunker changes, so ordinals aren't stable identities across runs. The
     * FK cascade handles the children. Wrapped in one transaction so a crash
     * mid-write can't leave a document indexed with a half-replaced chunk set —
     * which would silently corrupt retrieval in a way no test would catch.
     */
    @Transactional
    public long replaceDocument(SourceDoc doc, List<Chunk> chunks, List<float[]> embeddings) {
        if (chunks.size() != embeddings.size()) {
            throw new IllegalArgumentException(
                    "chunk/embedding count mismatch: " + chunks.size() + " vs " + embeddings.size());
        }

        jdbc.update("DELETE FROM document WHERE source_path = ? AND doc_version = ?",
                doc.sourcePath(), doc.docVersion());

        Long docId = jdbc.queryForObject("""
                INSERT INTO document (source_path, url, title, doc_version, content_hash)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """, Long.class,
                doc.sourcePath(), doc.url(), doc.title(), doc.docVersion(), doc.contentHash());

        List<Object[]> batch = new ArrayList<>(chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            Chunk c = chunks.get(i);
            batch.add(new Object[]{
                    docId, c.ordinal(), c.headingPath(), c.anchor(),
                    c.text(), c.tokenCount(), toVectorLiteral(embeddings.get(i))
            });
        }

        // The ::vector cast is required — pgvector has no JDBC type, so the
        // literal arrives as text and must be told what it is.
        jdbc.batchUpdate("""
            INSERT INTO chunk
                (document_id, ordinal, heading_path, anchor, text, token_count, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?::vector)
        """, batch);

        return docId;
    }

    /** pgvector's text format: [0.1,0.2,0.3] — no spaces, full precision. */
    static String toVectorLiteral(float[] v) {
        StringJoiner j = new StringJoiner(",", "[", "]");
        for (float f : v) j.add(Float.toString(f));
        return j.toString();
    }

    // ---- job queue -------------------------------------------------------

    public void enqueue(List<String> sourcePaths, String docVersion) {
        List<Object[]> batch = sourcePaths.stream()
                .map(p -> new Object[]{p, docVersion})
                .toList();
        jdbc.batchUpdate(
                "INSERT INTO ingest_job (source_path, doc_version) VALUES (?, ?)", batch);
    }

    /**
     * Claim a batch of jobs with SKIP LOCKED.
     *
     * <p>Two things this buys: concurrent workers never collide on the same row
     * without any external coordination, and the stale-lock clause reclaims work
     * abandoned by a crashed worker. Without that second condition a crash
     * mid-ingest strands those documents in RUNNING forever, and you'd only find
     * out via a quietly incomplete index.
     */
    @Transactional
    public List<ClaimedJob> claimBatch(int limit, String lockTimeout) {
        return jdbc.query("""
                UPDATE ingest_job
                SET state = 'RUNNING', locked_at = now(),
                    attempts = attempts + 1, updated_at = now()
                WHERE id IN (
                    SELECT id FROM ingest_job
                    WHERE state = 'PENDING'
                       OR (state = 'RUNNING' AND locked_at < now() - CAST(? AS INTERVAL))
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED
                    LIMIT ?
                )
                RETURNING id, source_path, doc_version, attempts
                """,
                (rs, n) -> new ClaimedJob(
                        rs.getLong("id"),
                        rs.getString("source_path"),
                        rs.getString("doc_version"),
                        rs.getInt("attempts")),
                lockTimeout, limit);
    }

    public void markDone(long jobId) {
        jdbc.update(
                "UPDATE ingest_job SET state = 'DONE', updated_at = now() WHERE id = ?", jobId);
    }

    /** Retry until max attempts, then park as FAILED with the error preserved. */
    public void markFailed(long jobId, String error, int maxAttempts) {
        jdbc.update("""
                UPDATE ingest_job
                SET state = CASE WHEN attempts >= ? THEN 'FAILED' ELSE 'PENDING' END,
                    last_error = ?, locked_at = NULL, updated_at = now()
                WHERE id = ?
                """, maxAttempts, truncate(error, 4000), jobId);
    }

    public int pendingCount() {
        Integer n = jdbc.queryForObject(
                "SELECT count(*) FROM ingest_job WHERE state IN ('PENDING','RUNNING')",
                Integer.class);
        return n == null ? 0 : n;
    }

    /**
     * Build the HNSW index after the bulk load.
     *
     * <p>Dropped before ingestion and rebuilt after: maintaining the graph
     * across tens of thousands of individual inserts is dramatically slower
     * than building it once over a populated table.
     */
    public void dropVectorIndex() {
        jdbc.execute("DROP INDEX IF EXISTS chunk_embedding_hnsw_idx");
    }

    public void buildVectorIndex() {
        jdbc.execute(
                "CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw_idx "
                        + "ON chunk USING hnsw (embedding vector_cosine_ops)");
        jdbc.execute("ANALYZE chunk");
    }

    private static String truncate(String s, int max) {
        if (s == null) return null;
        return s.length() <= max ? s : s.substring(0, max);
    }

    public record ClaimedJob(long id, String sourcePath, String docVersion, int attempts) {}
}
