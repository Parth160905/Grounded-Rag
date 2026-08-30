package dev.pgdocs.rag.ingest;

import java.util.List;

/**
 * Domain records for the ingestion path.
 *
 * <p>These are deliberately plain — no JPA entities, no annotations. The write
 * path is bulk inserts of a fixed shape; an ORM buys nothing here and its
 * caching would actively get in the way during a few-thousand-row batch load.
 */
public final class Documents {

    private Documents() {}

    /** A single source HTML page from the docs tarball. */
    public record SourceDoc(
            String sourcePath,   // 'sql-select.html'
            String url,          // canonical postgresql.org URL
            String title,
            String docVersion,
            String contentHash   // sha256 of raw HTML
    ) {}

    /**
     * One retrieval unit.
     *
     * @param headingPath breadcrumb like "SQL Commands > SELECT > Parameters".
     *                    Weighted above body text in the tsvector, and shown in
     *                    the source panel so a user can see where an answer came
     *                    from without opening the doc.
     * @param anchor      in-page anchor ('#SQL-FROM') for deep links. Null when
     *                    the section has none.
     * @param preserved   true when this chunk exceeded the token budget but was
     *                    kept whole anyway — a synopsis block, mostly. Tracked
     *                    because these are the chunks most likely to embed
     *                    poorly, and you want to be able to find them later.
     */
    public record Chunk(
            int ordinal,
            String headingPath,
            String anchor,
            String text,
            int tokenCount,
            boolean preserved
    ) {}

    /** A parsed document plus its chunks, ready to embed and persist. */
    public record ParsedDoc(SourceDoc doc, List<Chunk> chunks) {}
}
    

