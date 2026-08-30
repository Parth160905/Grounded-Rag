package dev.pgdocs.rag.ingest;

import dev.pgdocs.rag.ingest.Documents.Chunk;
import dev.pgdocs.rag.ingest.Documents.SourceDoc;
import dev.pgdocs.rag.ingest.IngestRepository.ClaimedJob;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

@Service
public class IngestService {

    private static final Logger log = LoggerFactory.getLogger(IngestService.class);

    private final IngestRepository repo;
    private final StructureAwareChunker chunker;
    private final EmbeddingService embeddings;

    private final Path docsDir;
    private final String docVersion;
    private final String baseUrl;
    private final int workerCount;
    private final int batchSize;
    private final int maxAttempts;
    private final Duration lockTimeout;

    public IngestService(
            IngestRepository repo,
            StructureAwareChunker chunker,
            EmbeddingService embeddings,
            @Value("${ingest.docs-dir}") String docsDir,
            @Value("${ingest.doc-version}") String docVersion,
            @Value("${ingest.base-url}") String baseUrl,
            @Value("${ingest.worker-count}") int workerCount,
            @Value("${ingest.batch-size}") int batchSize,
            @Value("${ingest.max-attempts}") int maxAttempts,
            @Value("${ingest.lock-timeout}") Duration lockTimeout) {
        this.repo = repo;
        this.chunker = chunker;
        this.embeddings = embeddings;
        this.docsDir = Path.of(docsDir);
        this.docVersion = docVersion;
        this.baseUrl = baseUrl;
        this.workerCount = workerCount;
        this.batchSize = batchSize;
        this.maxAttempts = maxAttempts;
        this.lockTimeout = lockTimeout;
    }

    /**
     * Full ingestion run: enqueue every page, then drain the queue in parallel.
     *
     * <p>Index handling brackets the run — dropped before, rebuilt after — so
     * the HNSW graph is built once over a populated table instead of being
     * maintained across tens of thousands of inserts.
     */
    public IngestReport ingestAll() throws IOException {
        List<String> pages = listPages();
        log.info("enqueueing {} pages (version {})", pages.size(), docVersion);
        repo.enqueue(pages, docVersion);

        repo.dropVectorIndex();
        long start = System.currentTimeMillis();

        AtomicInteger done = new AtomicInteger();
        AtomicInteger skipped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        AtomicInteger chunksWritten = new AtomicInteger();

        // Virtual threads: each worker spends nearly all its time blocked on
        // file I/O, ONNX inference, and JDBC. Platform threads would sit idle
        // holding a full stack each; this is the workload virtual threads exist
        // for, and it means no reactive rewrite later.
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < workerCount; i++) {
                pool.submit(() -> drain(done, skipped, failed, chunksWritten));
            }
            pool.shutdown();
            if (!pool.awaitTermination(2, TimeUnit.HOURS)) {
                log.warn("ingestion timed out with {} jobs outstanding", repo.pendingCount());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        log.info("building HNSW index over populated table");
        repo.buildVectorIndex();

        return new IngestReport(
                pages.size(), done.get(), skipped.get(), failed.get(),
                chunksWritten.get(), System.currentTimeMillis() - start);
    }

    private void drain(AtomicInteger done, AtomicInteger skipped,
                       AtomicInteger failed, AtomicInteger chunksWritten) {
        String timeout = lockTimeout.toSeconds() + " seconds";
        while (true) {
            List<ClaimedJob> batch = repo.claimBatch(batchSize, timeout);
            if (batch.isEmpty()) return;

            for (ClaimedJob job : batch) {
                try {
                    int written = process(job);
                    if (written < 0) {
                        skipped.incrementAndGet();
                    } else {
                        chunksWritten.addAndGet(written);
                        done.incrementAndGet();
                    }
                    repo.markDone(job.id());
                } catch (Exception e) {
                    log.warn("ingest failed for {} (attempt {}): {}",
                            job.sourcePath(), job.attempts(), e.toString());
                    repo.markFailed(job.id(), e.toString(), maxAttempts);
                    if (job.attempts() >= maxAttempts) failed.incrementAndGet();
                }
            }
        }
    }

    /** @return chunks written, or -1 if the document was unchanged and skipped. */
    private int process(ClaimedJob job) throws IOException {
        Path file = docsDir.resolve(job.sourcePath());
        String raw = Files.readString(file, StandardCharsets.UTF_8);
        String hash = sha256(raw);

        if (repo.existingHash(job.sourcePath(), job.docVersion())
                .filter(hash::equals).isPresent()) {
            return -1;
        }

        Document html = Jsoup.parse(raw, baseUrl);
        String title = titleOf(html, job.sourcePath());

        List<Chunk> chunks = chunker.chunk(html, title);
        if (chunks.isEmpty()) {
            log.debug("no chunks produced for {}", job.sourcePath());
            return 0;
        }

        List<float[]> vectors = embeddings.embedAll(chunks.stream().map(Chunk::text).toList());

        SourceDoc doc = new SourceDoc(
                job.sourcePath(), baseUrl + job.sourcePath(), title, job.docVersion(), hash);
        repo.replaceDocument(doc, chunks, vectors);

        log.debug("{} -> {} chunks", job.sourcePath(), chunks.size());
        return chunks.size();
    }

    /**
     * Every HTML file except the navigation scaffolding.
     *
     * <p>bookindex and the TOC pages are pure link lists. Indexing them adds
     * chunks that match many queries on keyword overlap while containing no
     * answer — precision poison, and exactly the kind of thing that makes a
     * retrieval system look mysteriously worse than it should.
     */
    private List<String> listPages() throws IOException {
        try (Stream<Path> files = Files.list(docsDir)) {
            return files
                    .filter(p -> p.getFileName().toString().endsWith(".html"))
                    .map(p -> p.getFileName().toString())
                    .filter(name -> !name.equals("bookindex.html"))
                    .filter(name -> !name.equals("index.html"))
                    .filter(name -> !name.startsWith("indexterm"))
                    .sorted()
                    .toList();
        }
    }

    private String titleOf(Document html, String fallback) {
        var h = html.selectFirst("h1, h2, title");
        return h == null ? fallback : h.text().strip();
    }

    static String sha256(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    public record IngestReport(
            int enqueued, int ingested, int skipped, int failed,
            int chunksWritten, long elapsedMs) {}
}
