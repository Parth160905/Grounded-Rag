-- Grounded RAG over the PostgreSQL 16 documentation
-- Flyway migration V1: core schema
--
-- Embedding dimension is pinned to 384 (bge-small-en-v1.5 / all-MiniLM-L6-v2).
-- Changing the embedding model requires a new migration + full re-embed.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Source documents
-- ---------------------------------------------------------------------------

CREATE TABLE document (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path   TEXT        NOT NULL,          -- 'sql-select.html'
    url           TEXT        NOT NULL,          -- canonical postgresql.org URL
    title         TEXT        NOT NULL,
    doc_version   TEXT        NOT NULL,          -- '16'
    content_hash  TEXT        NOT NULL,          -- sha256 of raw HTML; skip re-ingest if unchanged
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT document_source_version_uk UNIQUE (source_path, doc_version)
);

-- ---------------------------------------------------------------------------
-- Chunks: the retrieval unit
-- ---------------------------------------------------------------------------

CREATE TABLE chunk (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id   BIGINT      NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    ordinal       INT         NOT NULL,          -- position within document
    heading_path  TEXT        NOT NULL,          -- 'SQL Commands > SELECT > Parameters'
    anchor        TEXT,                          -- '#SQL-FROM', for deep links in the UI
    text          TEXT        NOT NULL,
    token_count   INT         NOT NULL,
    embedding     vector(384),

    -- Lexical channel. Heading path weighted above body text: a query naming a
    -- command should rank that command's own section over passing mentions.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(text, '')),         'B')
    ) STORED,

    CONSTRAINT chunk_document_ordinal_uk UNIQUE (document_id, ordinal)
);

-- Dense channel. Build AFTER the initial bulk embed: HNSW builds far faster on
-- a populated table than it maintains through a few thousand inserts.
CREATE INDEX chunk_embedding_hnsw_idx
    ON chunk USING hnsw (embedding vector_cosine_ops);

CREATE INDEX chunk_tsv_gin_idx ON chunk USING gin (tsv);
CREATE INDEX chunk_document_id_idx ON chunk (document_id);

-- ---------------------------------------------------------------------------
-- Ingestion queue
--
-- Postgres-as-queue via FOR UPDATE SKIP LOCKED. No Redis, no Kafka. Workers
-- claim rows with:
--
--   UPDATE ingest_job SET state = 'RUNNING', locked_at = now(), attempts = attempts + 1
--   WHERE id IN (
--       SELECT id FROM ingest_job
--       WHERE state = 'PENDING' OR (state = 'RUNNING' AND locked_at < now() - INTERVAL '10 minutes')
--       ORDER BY id
--       FOR UPDATE SKIP LOCKED
--       LIMIT 32
--   )
--   RETURNING *;
-- ---------------------------------------------------------------------------

CREATE TABLE ingest_job (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path  TEXT        NOT NULL,
    doc_version  TEXT        NOT NULL,
    state        TEXT        NOT NULL DEFAULT 'PENDING',
    attempts     INT         NOT NULL DEFAULT 0,
    last_error   TEXT,
    locked_at    TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT ingest_job_state_ck
        CHECK (state IN ('PENDING', 'RUNNING', 'DONE', 'FAILED'))
);

-- Partial index: the claim query only ever scans PENDING/RUNNING rows, and the
-- DONE rows will vastly outnumber them once ingestion settles.
CREATE INDEX ingest_job_claimable_idx
    ON ingest_job (id)
    WHERE state IN ('PENDING', 'RUNNING');

-- ---------------------------------------------------------------------------
-- API keys
-- ---------------------------------------------------------------------------

CREATE TABLE api_key (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    key_hash            TEXT        NOT NULL UNIQUE,   -- sha256(raw key); never store the raw key
    label               TEXT        NOT NULL,
    rate_limit_per_min  INT         NOT NULL DEFAULT 20,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at          TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Query log
--
-- This is the observability spine: it backs the latency/cost/groundedness
-- numbers you report, and lets you find real answers that scored badly and
-- promote them into the golden set.
-- ---------------------------------------------------------------------------

CREATE TABLE query_log (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    api_key_id          BIGINT REFERENCES api_key(id) ON DELETE SET NULL,
    question            TEXT        NOT NULL,
    retrieved_chunk_ids BIGINT[]    NOT NULL DEFAULT '{}',
    answer              TEXT,

    -- Verifier output
    claims_total        INT,
    claims_entailed     INT,
    groundedness        REAL,       -- claims_entailed / claims_total

    provider            TEXT,       -- 'ollama:qwen2.5:7b', 'groq:...'
    retrieval_ms        INT,
    generation_ms       INT,
    verification_ms     INT,
    tokens_in           INT,
    tokens_out          INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX query_log_created_at_idx ON query_log (created_at DESC);

-- Find weak answers fast — these are your golden-set candidates.
CREATE INDEX query_log_low_groundedness_idx
    ON query_log (groundedness)
    WHERE groundedness < 0.8;

-- ---------------------------------------------------------------------------
-- Note on the golden set
--
-- The 150-200 labelled question -> chunk pairs deliberately live in the repo as
-- JSON (eval/golden-set.json), NOT in this database. They need to be
-- version-controlled, diffable in review, and runnable in CI against a fresh
-- Testcontainers Postgres. A label set that only exists in someone's local DB
-- is not an artifact anyone can trust.
-- ---------------------------------------------------------------------------