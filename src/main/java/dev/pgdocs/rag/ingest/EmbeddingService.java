package dev.pgdocs.rag.ingest;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Embeddings, computed in-process via ONNX Runtime.
 *
 * <p>No API key, no per-token cost, no network. Three consequences worth being
 * explicit about, because they're the reason for the choice:
 *
 * <ul>
 *   <li>The eval suite is free to run, so you can iterate on retrieval as often
 *       as you like rather than rationing runs against a budget.</li>
 *   <li>Results are deterministic — the same corpus embeds identically every
 *       time, so a metric change is a real change and not provider drift.</li>
 *   <li>A stranger can clone the repo and ingest with no credentials, which is
 *       what makes the three-command setup claim actually true.</li>
 * </ul>
 *
 * <p>Cost is CPU: expect ingestion of the full docs set to take single-digit
 * minutes on a laptop rather than seconds.
 */
@Service
public class EmbeddingService {

    public static final int DIMENSION = 384;

    private final EmbeddingModel model;

    public EmbeddingService(EmbeddingModel model) {
        this.model = model;
    }

    public List<float[]> embedAll(List<String> texts) {
        List<TextSegment> segments = texts.stream().map(TextSegment::from).toList();
        return model.embedAll(segments).content().stream()
                .map(e -> {
                    float[] v = e.vector();
                    if (v.length != DIMENSION) {
                        // Guard against a model swap silently mismatching the
                        // schema — the insert would fail far from the cause.
                        throw new IllegalStateException(
                                "expected dim " + DIMENSION + ", got " + v.length);
                    }
                    return v;
                })
                .toList();
    }

    public float[] embedOne(String text) {
        return embedAll(List.of(text)).getFirst();
    }

    @Configuration
    static class ModelConfig {
        /**
         * Loaded once at startup and shared. The model is thread-safe for
         * inference, so all ingestion workers use this single instance rather
         * than each paying the load cost.
         */
        @Bean
        EmbeddingModel embeddingModel() {
            return new BgeSmallEnV15EmbeddingModel();
        }
    }
}
