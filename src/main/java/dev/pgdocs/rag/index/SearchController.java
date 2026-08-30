package dev.pgdocs.rag.index;

import dev.pgdocs.rag.index.HybridSearchRepository.Hit;
import dev.pgdocs.rag.ingest.EmbeddingService;
import dev.pgdocs.rag.ingest.IngestService;
import dev.pgdocs.rag.ingest.IngestService.IngestReport;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api")
public class SearchController {

    private final HybridSearchRepository search;
    private final EmbeddingService embeddings;
    private final IngestService ingest;

    public SearchController(HybridSearchRepository search,
                            EmbeddingService embeddings,
                            IngestService ingest) {
        this.search = search;
        this.embeddings = embeddings;
        this.ingest = ingest;
    }

    /**
     * Milestone 1 lives here: hybrid search reachable from curl, before any
     * generation exists. Get this returning sane results and the hard part of
     * the system is done — a RAG pipeline is only ever as good as what
     * retrieval hands it.
     *
     * <pre>
     * curl -s "localhost:8080/api/search?q=how+do+I+cancel+a+long+running+query" | jq
     * </pre>
     */
    @GetMapping("/search")
    public SearchResponse search(
            @RequestParam("q") @NotBlank String query,
            @RequestParam(defaultValue = "8") @Min(1) @Max(50) int limit,
            @RequestParam(defaultValue = "50") @Min(1) @Max(200) int candidates) {

        long start = System.nanoTime();
        float[] vector = embeddings.embedOne(query);
        long embedMs = (System.nanoTime() - start) / 1_000_000;

        long searchStart = System.nanoTime();
        List<Hit> hits = search.search(query, vector, limit, candidates);
        long searchMs = (System.nanoTime() - searchStart) / 1_000_000;

        return new SearchResponse(
                query,
                hits.stream().map(SearchResult::from).toList(),
                embedMs, searchMs);
    }

    /** Admin-only in anything real. Fine unguarded while you're building. */
    @PostMapping("/admin/ingest")
    public IngestReport ingest() throws IOException {
        return ingest.ingestAll();
    }

    public record SearchResponse(
            String query, List<SearchResult> results, long embedMs, long searchMs) {}

    public record SearchResult(
            long chunkId,
            String docTitle,
            String headingPath,
            String deepLink,
            String snippet,
            int tokenCount,
            double rrfScore,
            Integer denseRank,
            Integer lexicalRank
    ) {
        static SearchResult from(Hit h) {
            return new SearchResult(
                    h.chunkId(), h.docTitle(), h.headingPath(), h.deepLink(),
                    h.text().length() > 400 ? h.text().substring(0, 400) + "…" : h.text(),
                    h.tokenCount(), h.rrfScore(), h.denseRank(), h.lexicalRank());
        }
    }
}
