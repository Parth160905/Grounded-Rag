# Grounded RAG

Hybrid retrieval over the PostgreSQL 16 documentation. BM25 and dense vector
search run independently, then fuse with reciprocal rank fusion. Every result
carries a heading path and a deep link back to the source section.

**recall@5 0.819 · MRR 0.836** on a 12-query gold set, enforced as a build gate.

## Why hybrid

Neither retriever is sufficient alone, and the failure modes differ.

Exact identifiers (`pg_hba.conf`, `work_mem`, `EXPLAIN ANALYZE`) are BM25's
strength — the dense model often ranks them poorly because the surface form
carries the meaning. Symptom phrasings ("queries are slow", "why is my
connection being refused") are the reverse: BM25 has nothing to match, while
the embedding finds the right section.

On *deadlock detected*, the top result is dense rank 1 / lexical rank 8 and the
second is dense 10 / lexical 1 — near-total disagreement, both correct, both
surfaced. The UI labels each result with its per-channel ranks.

## Architecture

Chunks retain their heading path (`13.3. Explicit Locking > 13.3.4. Deadlocks`)
and anchor, which is what makes the deep links work.

The fusion is a single SQL statement. Both retrievers rank independently, a
full outer join keeps rows found by only one, and misses score as zero:

    COALESCE(1.0 / (? + d.rank), 0.0) + COALESCE(1.0 / (? + l.rank), 0.0)

Per-channel ranks are returned alongside the fused score. When a query returns
something wrong, that field says which channel pulled it in.

## Stack

- Java 21, Spring Boot 3.3
- PostgreSQL 16 + pgvector, Flyway migrations
- Embeddings in-process via ONNX
- Ollama for generation only
- No frontend build step, one static HTML file

## Evaluation

`RetrievalEvalTest` calls the repository directly, scores 12 gold queries, and
fails the build if quality regresses (MRR >= 0.80, recall@5 >= 0.78).

Failures print the missed chunk IDs plus the top-5 returned with their dense and
lexical ranks, so a red build says which channel degraded.

The gold set spans three query shapes deliberately: exact identifiers, task
phrasings, and symptom phrasings. Labels were written by inspecting actual
top-10 output per query.

## Running it

    docker compose up -d postgres
    JAVA_HOME=$(/usr/libexec/java_home -v 21) ./gradlew bootRun

Then open http://localhost:8080

    ./gradlew test
    curl -s "localhost:8080/api/search?q=deadlock+detected" | jq

The container-backed ingest test is tagged `integration` and excluded from the
default test task; it needs a Docker daemon.

## Performance

Warm, single node, 8-result page: embedding ~14ms, search ~8ms.

## Limitations

- Adjacent chunks from the same section can occupy multiple top-k slots.
- Release-note sections sometimes surface for symptom queries.
- RRF ties are broken arbitrarily when both channels return identical ranks.
- 12 gold queries is small. The numbers are honest but the interval is wide.
