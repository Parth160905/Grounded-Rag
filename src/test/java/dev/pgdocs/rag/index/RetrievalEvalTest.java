package dev.pgdocs.rag.index;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.pgdocs.rag.ingest.EmbeddingService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
class RetrievalEvalTest {

    private static final int K = 5;
    private static final int LIMIT = 8;
    private static final int CANDIDATES = 50;

    @Autowired EmbeddingService embeddings;
    @Autowired HybridSearchRepository search;

    @Test
    void retrievalQualityMeetsBaseline() throws Exception {
        JsonNode gold = new ObjectMapper()
                .readTree(new ClassPathResource("gold.json").getInputStream())
                .get("queries");

        double recallSum = 0, rrSum = 0;
        int n = gold.size();

        System.out.printf("%n%9s %6s  %s%n", "recall@" + K, "RR", "query");
        System.out.println("-".repeat(72));

        for (JsonNode g : gold) {
            String q = g.get("q").asText();

            Set<Long> relevant = new HashSet<>();
            g.get("relevant").forEach(v -> relevant.add(v.asLong()));

            List<HybridSearchRepository.Hit> hits =
                    search.search(q, embeddings.embedOne(q), LIMIT, CANDIDATES);

            List<Long> ids = hits.stream().limit(K)
                    .map(HybridSearchRepository.Hit::chunkId).toList();

            long found = ids.stream().filter(relevant::contains).count();
            double recall = relevant.isEmpty() ? 0 : (double) found / relevant.size();

            double rr = 0;
            for (int i = 0; i < ids.size(); i++) {
                if (relevant.contains(ids.get(i))) { rr = 1.0 / (i + 1); break; }
            }

            recallSum += recall;
            rrSum += rr;
            System.out.printf("%9.2f %6.2f  %s%n", recall, rr,
                    q.length() > 45 ? q.substring(0, 45) : q);

            if (recall < 1.0) {
                System.out.println("           missed: " + diff(relevant, ids));
                hits.stream().limit(K).forEach(h -> System.out.printf(
                        "           got %d  d=%s l=%s  %s%n",
                        h.chunkId(), h.denseRank(), h.lexicalRank(),
                        trim(h.headingPath())));
            }
        }

        double recallAtK = recallSum / n, mrr = rrSum / n;
        System.out.println("-".repeat(72));
        System.out.printf("recall@%d: %.3f   MRR: %.3f   (n=%d)%n%n", K, recallAtK, mrr, n);

        assertTrue(mrr >= 0.60, "MRR regressed: " + mrr);
        assertTrue(recallAtK >= 0.50, "recall@" + K + " regressed: " + recallAtK);
    }

    private static String diff(Set<Long> relevant, List<Long> got) {
        List<Long> missed = new ArrayList<>(relevant);
        missed.removeAll(got);
        return missed.toString();
    }

    private static String trim(String s) {
        return s == null ? "" : s.length() > 55 ? s.substring(0, 55) : s;
    }
}
