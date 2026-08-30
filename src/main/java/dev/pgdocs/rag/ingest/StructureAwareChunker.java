package dev.pgdocs.rag.ingest;

import dev.pgdocs.rag.ingest.Documents.Chunk;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a PostgreSQL docs HTML page along its own section hierarchy.
 *
 * <p>Why not fixed-size splitting: the Postgres docs are reference material
 * where meaning is carried by structure. A fixed 500-token window will cut a
 * SELECT synopsis in half and orphan a parameter definition from the command it
 * documents. Both failures are invisible in the index and unrecoverable by
 * retrieval tuning — the chunk simply no longer contains a complete answer.
 *
 * <p>Four rules, in priority order:
 * <ol>
 *   <li><b>Split on section boundaries.</b> The docs use nested
 *       {@code div.sect1 / .sect2 / .sect3} with a heading as the first child.
 *       That hierarchy is the author's own semantic segmentation — better than
 *       anything inferred.</li>
 *   <li><b>Never split a synopsis.</b> A {@code <pre class="synopsis">} block is
 *       atomic: half a grammar production answers nothing. These are kept whole
 *       even when they blow the budget.</li>
 *   <li><b>Merge undersized sections into their parent.</b> A 20-token section
 *       standing alone is noise in the index — it matches on stray terms and
 *       carries no answer.</li>
 *   <li><b>Split oversized prose on paragraph boundaries, with heading
 *       repetition.</b> Every part re-states the heading path so a fragment is
 *       still self-describing when retrieved in isolation.</li>
 * </ol>
 */
@Component
public class StructureAwareChunker {

    private final int maxTokens;
    private final int minTokens;
    private final double charsPerToken;

    public StructureAwareChunker(
            @Value("${chunking.max-tokens}") int maxTokens,
            @Value("${chunking.min-tokens}") int minTokens,
            @Value("${chunking.chars-per-token}") double charsPerToken) {
        this.maxTokens = maxTokens;
        this.minTokens = minTokens;
        this.charsPerToken = charsPerToken;
    }

    public List<Chunk> chunk(Document html, String docTitle) {
        List<Chunk> out = new ArrayList<>();

        Element body = html.selectFirst("div.book, div.refentry, div.chapter, body");
        if (body == null) return out;

        // Top-level sections. refentry pages (the SQL command reference) use
        // refsect1; chapter pages use sect1. Handle both.
        Elements sections = body.select("> div.sect1, > div.refsect1, div.refentry > div.refsect1");
        if (sections.isEmpty()) {
            // Flat page with no sections — treat the whole body as one unit.
            emit(out, docTitle, null, extractText(body), false);
            return out;
        }

        for (Element section : sections) {
            walk(section, docTitle, out);
        }
        return out;
    }

    /** Depth-first walk. Leaf sections become chunks; parents recurse. */
    private void walk(Element section, String parentPath, List<Chunk> out) {
        String heading = headingOf(section);
        String path = heading == null ? parentPath : parentPath + " > " + heading;
        String anchor = anchorOf(section);

        Elements children = section.select("> div.sect2, > div.sect3, > div.refsect2, > div.refsect3");

        if (children.isEmpty()) {
            emitSection(out, path, anchor, section);
            return;
        }

        // A parent section often has its own intro prose before the first
        // subsection. That prose is real content — don't drop it. But if it's
        // too short to stand alone, rule 3 says let it go rather than pollute
        // the index with a fragment.
        String intro = introTextBefore(section, children.first());
        if (!intro.isBlank() && estimateTokens(intro) >= minTokens) {
            emit(out, path, anchor, intro, false);
        }

        for (Element child : children) {
            walk(child, path, out);
        }
    }

    /** Emit one section, applying the synopsis and size rules. */
    private void emitSection(List<Chunk> out, String path, String anchor, Element section) {
        Element synopsis = section.selectFirst("pre.synopsis");

        if (synopsis != null) {
            // Rule 2: the synopsis is atomic. Emit it whole with its heading,
            // regardless of length, flagged as preserved.
            String text = path + "\n\n" + synopsis.wholeText().strip();
            out.add(new Chunk(out.size(), path, anchor, text,
                    estimateTokens(text), true));

            // Surrounding prose becomes its own chunk if substantial enough.
            Element clone = section.clone();
            clone.select("pre.synopsis").remove();
            String prose = extractText(clone);
            if (estimateTokens(prose) >= minTokens) {
                emit(out, path, anchor, prose, false);
            }
            return;
        }

        emit(out, path, anchor, extractText(section), false);
    }

    /** Emit text, splitting on paragraphs if it exceeds the budget. */
    private void emit(List<Chunk> out, String path, String anchor, String text, boolean preserved) {
        text = text.strip();
        if (text.isBlank()) return;

        int tokens = estimateTokens(text);
        if (tokens <= maxTokens) {
            // Rule 3: drop fragments too small to carry an answer. The one
            // exception is a preserved synopsis, which is meaningful at any size.
            if (tokens < minTokens && !preserved) return;
            out.add(new Chunk(out.size(), path, anchor, withHeading(path, text),
                    tokens, preserved));
            return;
        }

        // Rule 4: paragraph-boundary split, heading repeated on every part.
        String[] paragraphs = text.split("\n{2,}");
        StringBuilder buf = new StringBuilder();
        for (String para : paragraphs) {
            int combined = estimateTokens(buf + "\n\n" + para);
            if (combined > maxTokens && !buf.isEmpty()) {
                flush(out, path, anchor, buf.toString());
                buf.setLength(0);
            }
            if (!buf.isEmpty()) buf.append("\n\n");
            buf.append(para);
        }
        if (!buf.isEmpty()) flush(out, path, anchor, buf.toString());
    }

    private void flush(List<Chunk> out, String path, String anchor, String text) {
        String body = text.strip();
        if (body.isBlank()) return;
        String full = withHeading(path, body);
        out.add(new Chunk(out.size(), path, anchor, full, estimateTokens(full), false));
    }

    /**
     * Prefix the heading path onto the chunk text.
     *
     * <p>This matters more than it looks. The embedding is computed over this
     * exact string, so including the breadcrumb pulls the vector toward the
     * section's topic — a chunk from "SELECT > Parameters" embeds near queries
     * about SELECT even when the body never repeats the word.
     */
    private String withHeading(String path, String text) {
        return text.startsWith(path) ? text : path + "\n\n" + text;
    }

    private String headingOf(Element section) {
        Element h = section.selectFirst("h1, h2, h3, h4, h5");
        return h == null ? null : h.text().strip();
    }

    private String anchorOf(Element section) {
        String id = section.id();
        if (id != null && !id.isBlank()) return "#" + id;
        Element anchored = section.selectFirst("[id]");
        return anchored == null ? null : "#" + anchored.id();
    }

    /** Text of a parent section up to (not including) its first subsection. */
    private String introTextBefore(Element section, Element firstChild) {
        Element clone = section.clone();
        clone.select("div.sect2, div.sect3, div.refsect2, div.refsect3").remove();
        clone.select("h1, h2, h3, h4, h5").remove();
        return extractText(clone);
    }

    /**
     * HTML to clean text, preserving block structure.
     *
     * <p>Tables are flattened to pipe-delimited rows rather than dropped —
     * the config-parameter and data-type reference tables carry a large share
     * of the answers in this corpus, and losing them would put a hole in
     * retrieval exactly where users ask the most questions.
     */
    private String extractText(Element el) {
        Element clone = el.clone();

        clone.select("table").forEach(table -> {
            StringBuilder sb = new StringBuilder();
            for (Element row : table.select("tr")) {
                Elements cells = row.select("th, td");
                if (cells.isEmpty()) continue;
                sb.append(String.join(" | ", cells.eachText())).append("\n");
            }
            table.replaceWith(new Element("p").text(sb.toString().strip()));
        });

        clone.select("p, div, li, pre, h1, h2, h3, h4, h5, br")
             .forEach(e -> e.after("\n\n"));

        return clone.wholeText()
                .replaceAll("[ \\t]+", " ")
                .replaceAll("\n{3,}", "\n\n")
                .strip();
    }

    /**
     * Rough token estimate from character count.
     *
     * <p>Deliberately cheap and deliberately conservative. Running the real
     * tokenizer per candidate split would dominate ingestion time, and the
     * budget already sits well under the model's 512 limit — so an estimate
     * that errs small costs a little index density and buys a lot of speed.
     */
    private int estimateTokens(String text) {
        return (int) Math.ceil(text.length() / charsPerToken);
    }
}
