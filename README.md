# pgdocs-rag

Hybrid retrieval over the PostgreSQL 16 documentation, built on Java 21 and Spring Boot.
Dense vectors and lexical search fused by reciprocal rank, with embeddings running
in-process via ONNX — no API key required to run it.

> **Status: ingestion + retrieval.** Generation, claim-level citation verification,
> and the eval suite are not built yet. The metrics table below is intentionally
> empty rather than filled with numbers that haven't been measured.

## Setup

```bash
docker compose up -d

curl -O https://ftp.postgresql.org/pub/docs/16/postgresql-16-A4.html.tar.gz
mkdir -p docs/pg16 && tar xzf postgresql-16-A4.html.tar.gz -C docs/pg16

./gradlew bootRun
curl -X POST localhost:8080/api/admin/ingest
```

Then:

```bash
curl -s "localhost:8080/api/search?q=how+do+I+cancel+a+long+running+query" | jq
```

## Design notes

**Structure-aware chunking.** Fixed-size splitting cuts SQL synopses in half and
orphans parameter definitions from the command they document. Both failures are
invisible in the index and unrecoverable by retrieval tuning. The chunker splits
on the docs' own `refsect`/`sect` hierarchy, keeps `pre.synopsis` blocks atomic
regardless of length, merges undersized sections into their parent, and repeats
the heading path onto every chunk so a fragment is self-describing when retrieved
alone.

**RRF over weighted score blending.** Cosine similarity and `ts_rank_cd` sit on
incomparable scales, so any fixed weighting is a magic number tuned to one query
shape. RRF fuses on rank position only and needs no calibration.

**Postgres as the queue.** `FOR UPDATE SKIP LOCKED` with a stale-lock timeout —
no Redis, no Kafka, and a crashed worker's jobs get reclaimed rather than
stranded in `RUNNING`.

**In-process embeddings.** bge-small-en-v15 via ONNX Runtime. Free to run, fully
deterministic across runs, and it means the eval suite can be iterated on without
rationing against an API budget.

## Results

Not yet measured. This table gets filled once the golden set and eval harness
exist, and every number in it will be reproducible by running `./gradlew evalTest`.

| Metric | Value |
|---|---|
| Recall@5 | — |
| MRR | — |
| Groundedness rate | — |
| Hallucinated-citation rate | — |
| p95 retrieval latency | — |

## Limitations

- **Token counts are estimated from character length**, not tokenized. Chunks
  near the budget may run over the model's 512-token window and be silently
  truncated at embed time. The budget is set conservatively to make this rare,
  not impossible.
- **The lexical channel is `ts_rank_cd`, not BM25.** Postgres full-text scoring
  differs from true BM25 in how it handles term saturation and length
  normalisation. Adequate as a fusion channel, but a Lucene index would be
  better and is the obvious v2.
- **Version-specific content is not disambiguated.** Only the 16 docs are
  ingested. Questions whose answer changed across versions will be answered from
  16 without flagging that the behaviour differs elsewhere — a known hard class
  the eval set should target.
- **No reranker.** A cross-encoder over the fused top-k would likely improve
  precision@5 meaningfully; not yet measured, so not yet claimed.
- **Table flattening loses column semantics.** Reference tables are collapsed to
  pipe-delimited rows, which preserves the content but discards header-to-cell
  relationships.

## License

The PostgreSQL documentation is redistributed under its own license — confirm the
current terms and include the notice before publishing this repository.`