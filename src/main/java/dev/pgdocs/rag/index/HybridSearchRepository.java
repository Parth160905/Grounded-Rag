package dev.pgdocs.rag.index;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.StringJoiner;

/**
 * Hybrid retrieval: dense vectors and lexical search, fused by reciprocal rank.
 *
 * <p>Neither channel is sufficient alone on this corpus. Dense retrieval handles
 * paraphrase — "how do I stop a query from running forever" finds
 * {@code statement_timeout} without sharing a word with it. Lexical retrieval
 * handles exact identifiers, which is what users actually type: a vector search
 * for {@code pg_stat_activity} will happily return semantically adjacent
 * monitoring views instead of the one that was named.
 *
 * <p><b>Why RRF rather than weighted score blending.</b> Cosine similarity and
 * {@code ts_rank} are on incomparable scales with different distributions, so
 * any fixed weighting is a magic number tuned to one query shape and wrong for
 * the next. RRF discards the scores and fuses on rank position alone, which
 * needs no calibration and is what you can defend in an interview.
 */
@Repository
public class HybridSearchRepository {

    /**
     * RRF smoothing constant. 60 is the value from the original Cormack et al.
     * work and is the standard default; it damps the influence of top-1
     * positions enough that one channel being confidently wrong doesn't
     * dominate the fused list.
     */
    private static final int RRF_K = 60;

    private final JdbcTemplate jdbc;

    public HybridSearchRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /**
     * @param limit      final result count
     * @param candidates per-channel depth before fusion. Wider than {@code limit}
     *                   on purpose — a chunk ranked 30th by vectors and 3rd by
     *                   text should surface, and it can't if each channel only
     *                   contributes its own top few.
     */
    public List<Hit> search(String query, float[] queryVector, int limit, int candidates) {
        return jdbc.query("""
                WITH dense AS (
                    SELECT id,
                           row_number() OVER (ORDER BY embedding <=> CAST(? AS vector)) AS rank
                    FROM chunk
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(? AS vector)
                    LIMIT ?
                ),
                lexical AS (
                    SELECT id,
                           row_number() OVER (
                               ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', ?)) DESC
                           ) AS rank
                    FROM chunk
                    WHERE tsv @@ websearch_to_tsquery('english', ?)
                    LIMIT ?
                ),
                fused AS (
                    SELECT COALESCE(d.id, l.id) AS id,
                           COALESCE(1.0 / (? + d.rank), 0.0)
                         + COALESCE(1.0 / (? + l.rank), 0.0) AS rrf_score,
                           d.rank AS dense_rank,
                           l.rank AS lexical_rank
                    FROM dense d
                    FULL OUTER JOIN lexical l ON d.id = l.id
                )
                SELECT f.id, f.rrf_score, f.dense_rank, f.lexical_rank,
                       c.heading_path, c.anchor, c.text, c.token_count,
                       doc.title, doc.url
                FROM fused f
                JOIN chunk c ON c.id = f.id
                JOIN document doc ON doc.id = c.document_id
                ORDER BY f.rrf_score DESC
                LIMIT ?
                """,
                (rs, n) -> new Hit(
                        rs.getLong("id"),
                        rs.getDouble("rrf_score"),
                        (Integer) rs.getObject("dense_rank"),
                        (Integer) rs.getObject("lexical_rank"),
                        rs.getString("heading_path"),
                        rs.getString("anchor"),
                        rs.getString("text"),
                        rs.getInt("token_count"),
                        rs.getString("title"),
                        rs.getString("url")),
                toVectorLiteral(queryVector), toVectorLiteral(queryVector), candidates,
                query, query, candidates,
                RRF_K, RRF_K,
                limit);
    }

    static String toVectorLiteral(float[] v) {
        StringJoiner j = new StringJoiner(",", "[", "]");
        for (float f : v) j.add(Float.toString(f));
        return j.toString();
    }

    /**
     * Per-channel ranks are returned deliberately, not just the fused score.
     * When a query returns something wrong you need to know which channel
     * pulled it in — that single field is the difference between debugging
     * retrieval and guessing at it, and it's what the eval failure reports
     * will key off.
     */
    public record Hit(
            long chunkId,
            double rrfScore,
            Integer denseRank,
            Integer lexicalRank,
            String headingPath,
            String anchor,
            String text,
            int tokenCount,
            String docTitle,
            String docUrl
    ) {
        public String deepLink() {
            return anchor == null ? docUrl : docUrl + anchor;
        }
    }
}