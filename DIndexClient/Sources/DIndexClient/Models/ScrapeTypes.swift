import Foundation

// MARK: - Scrape Options

/// Options for web scraping operations
public struct ScrapeOptions: Codable, Sendable {
    /// Maximum crawl depth (0 = current page only, default: 2)
    public var maxDepth: UInt8

    /// Stay on the same domain (default: true)
    public var stayOnDomain: Bool

    /// Delay between requests in milliseconds (default: 1000)
    public var delayMs: UInt64

    /// Maximum number of pages to scrape (default: 100)
    public var maxPages: Int

    enum CodingKeys: String, CodingKey {
        case maxDepth = "max_depth"
        case stayOnDomain = "stay_on_domain"
        case delayMs = "delay_ms"
        case maxPages = "max_pages"
    }

    public init(
        maxDepth: UInt8 = 2,
        stayOnDomain: Bool = true,
        delayMs: UInt64 = 1000,
        maxPages: Int = 100
    ) {
        self.maxDepth = maxDepth
        self.stayOnDomain = stayOnDomain
        self.delayMs = delayMs
        self.maxPages = maxPages
    }
}

// MARK: - Scrape Request

/// Request to start a web scrape job
public struct ScrapeRequest: Codable, Sendable {
    /// URLs to start scraping from
    public let urls: [String]

    /// Scrape options
    public let options: ScrapeOptions

    public init(urls: [String], options: ScrapeOptions = ScrapeOptions()) {
        self.urls = urls
        self.options = options
    }
}

// MARK: - Job Responses

/// Response when a job is started
public struct JobStartedResponse: Codable, Sendable {
    /// The job ID
    public let jobId: String

    /// Human-readable message
    public let message: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case message
    }
}

/// Response for job progress queries
public struct JobProgress: Codable, Sendable {
    /// The job ID
    public let jobId: String

    /// Current stage of the job (e.g., "scraping", "indexing", "completed", "failed: ...")
    public let stage: String

    /// Current progress count
    public let current: UInt64

    /// Total count (if known)
    public let total: UInt64?

    /// Processing rate (items/sec)
    public let rate: Double?

    /// Estimated time remaining in seconds
    public let etaSeconds: UInt64?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case stage
        case current
        case total
        case rate
        case etaSeconds = "eta_seconds"
    }

    /// Whether the job is completed
    public var isCompleted: Bool {
        stage == "completed"
    }

    /// Whether the job failed
    public var isFailed: Bool {
        stage.hasPrefix("failed:")
    }

    /// Whether the job is still running
    public var isRunning: Bool {
        !isCompleted && !isFailed
    }

    /// Error message if the job failed
    public var errorMessage: String? {
        guard isFailed else { return nil }
        return String(stage.dropFirst("failed: ".count))
    }

    /// Progress as a percentage (0.0 to 1.0)
    public var progress: Double {
        guard let total = total, total > 0 else { return 0 }
        return Double(current) / Double(total)
    }
}

/// Response for job cancellation
public struct JobCancelResponse: Codable, Sendable {
    /// Whether the cancellation was successful
    public let success: Bool

    /// Human-readable message
    public let message: String
}

// MARK: - SSE Event Types

/// How a URL entered the frontier
public enum UrlSource: String, Codable, Sendable {
    case seed
    case discovered
}

/// Statistics for completed jobs
public struct JobStats: Codable, Sendable {
    public let documentsProcessed: Int
    public let chunksIndexed: Int
    public let durationMs: UInt64
    public let errors: Int

    enum CodingKeys: String, CodingKey {
        case documentsProcessed = "documents_processed"
        case chunksIndexed = "chunks_indexed"
        case durationMs = "duration_ms"
        case errors
    }
}

/// SSE events emitted during scrape jobs
public enum ScrapeEvent: Codable, Sendable {
    /// Job has been created and seeds are queued
    case jobStarted(jobId: String, seedUrls: [String], maxDepth: UInt8, maxPages: Int)

    /// A URL was added to the crawl frontier
    case urlQueued(jobId: String, url: String, depth: UInt8, source: UrlSource)

    /// A URL fetch has begun
    case urlFetching(jobId: String, url: String)

    /// A URL was successfully fetched, extracted, and indexed
    case urlIndexed(jobId: String, url: String, title: String?, wordCount: Int, chunksCreated: Int, durationMs: UInt64, discoveredUrls: Int)

    /// A URL failed to fetch or extract
    case urlFailed(jobId: String, url: String, error: String, durationMs: UInt64)

    /// A URL was skipped (duplicate, robots.txt, etc.)
    case urlSkipped(jobId: String, url: String, reason: String)

    /// Aggregate progress snapshot emitted after each URL
    case progress(jobId: String, urlsProcessed: UInt64, urlsSucceeded: UInt64, urlsFailed: UInt64, urlsSkipped: UInt64, urlsQueued: Int, chunksIndexed: Int, elapsedMs: UInt64, rate: Double?, etaSeconds: UInt64?)

    /// Job finished (completed, failed, or cancelled)
    case jobCompleted(jobId: String, status: String, stats: JobStats?, error: String?)

    /// SSE lagged event (some events were missed)
    case lagged(missed: Int)

    enum CodingKeys: String, CodingKey {
        case type
        case jobId = "job_id"
        case seedUrls = "seed_urls"
        case maxDepth = "max_depth"
        case maxPages = "max_pages"
        case url
        case depth
        case source
        case title
        case wordCount = "word_count"
        case chunksCreated = "chunks_created"
        case durationMs = "duration_ms"
        case discoveredUrls = "discovered_urls"
        case error
        case reason
        case urlsProcessed = "urls_processed"
        case urlsSucceeded = "urls_succeeded"
        case urlsFailed = "urls_failed"
        case urlsSkipped = "urls_skipped"
        case urlsQueued = "urls_queued"
        case chunksIndexed = "chunks_indexed"
        case elapsedMs = "elapsed_ms"
        case rate
        case etaSeconds = "eta_seconds"
        case status
        case stats
        case missed
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "job_started":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let seedUrls = try container.decode([String].self, forKey: .seedUrls)
            let maxDepth = try container.decode(UInt8.self, forKey: .maxDepth)
            let maxPages = try container.decode(Int.self, forKey: .maxPages)
            self = .jobStarted(jobId: jobId, seedUrls: seedUrls, maxDepth: maxDepth, maxPages: maxPages)

        case "url_queued":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let url = try container.decode(String.self, forKey: .url)
            let depth = try container.decode(UInt8.self, forKey: .depth)
            let source = try container.decode(UrlSource.self, forKey: .source)
            self = .urlQueued(jobId: jobId, url: url, depth: depth, source: source)

        case "url_fetching":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let url = try container.decode(String.self, forKey: .url)
            self = .urlFetching(jobId: jobId, url: url)

        case "url_indexed":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let url = try container.decode(String.self, forKey: .url)
            let title = try container.decodeIfPresent(String.self, forKey: .title)
            let wordCount = try container.decode(Int.self, forKey: .wordCount)
            let chunksCreated = try container.decode(Int.self, forKey: .chunksCreated)
            let durationMs = try container.decode(UInt64.self, forKey: .durationMs)
            let discoveredUrls = try container.decode(Int.self, forKey: .discoveredUrls)
            self = .urlIndexed(jobId: jobId, url: url, title: title, wordCount: wordCount, chunksCreated: chunksCreated, durationMs: durationMs, discoveredUrls: discoveredUrls)

        case "url_failed":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let url = try container.decode(String.self, forKey: .url)
            let error = try container.decode(String.self, forKey: .error)
            let durationMs = try container.decode(UInt64.self, forKey: .durationMs)
            self = .urlFailed(jobId: jobId, url: url, error: error, durationMs: durationMs)

        case "url_skipped":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let url = try container.decode(String.self, forKey: .url)
            let reason = try container.decode(String.self, forKey: .reason)
            self = .urlSkipped(jobId: jobId, url: url, reason: reason)

        case "progress":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let urlsProcessed = try container.decode(UInt64.self, forKey: .urlsProcessed)
            let urlsSucceeded = try container.decode(UInt64.self, forKey: .urlsSucceeded)
            let urlsFailed = try container.decode(UInt64.self, forKey: .urlsFailed)
            let urlsSkipped = try container.decode(UInt64.self, forKey: .urlsSkipped)
            let urlsQueued = try container.decode(Int.self, forKey: .urlsQueued)
            let chunksIndexed = try container.decode(Int.self, forKey: .chunksIndexed)
            let elapsedMs = try container.decode(UInt64.self, forKey: .elapsedMs)
            let rate = try container.decodeIfPresent(Double.self, forKey: .rate)
            let etaSeconds = try container.decodeIfPresent(UInt64.self, forKey: .etaSeconds)
            self = .progress(jobId: jobId, urlsProcessed: urlsProcessed, urlsSucceeded: urlsSucceeded, urlsFailed: urlsFailed, urlsSkipped: urlsSkipped, urlsQueued: urlsQueued, chunksIndexed: chunksIndexed, elapsedMs: elapsedMs, rate: rate, etaSeconds: etaSeconds)

        case "job_completed":
            let jobId = try container.decode(String.self, forKey: .jobId)
            let status = try container.decode(String.self, forKey: .status)
            let stats = try container.decodeIfPresent(JobStats.self, forKey: .stats)
            let error = try container.decodeIfPresent(String.self, forKey: .error)
            self = .jobCompleted(jobId: jobId, status: status, stats: stats, error: error)

        case "lagged":
            let missed = try container.decode(Int.self, forKey: .missed)
            self = .lagged(missed: missed)

        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unknown event type: \(type)")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .jobStarted(let jobId, let seedUrls, let maxDepth, let maxPages):
            try container.encode("job_started", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(seedUrls, forKey: .seedUrls)
            try container.encode(maxDepth, forKey: .maxDepth)
            try container.encode(maxPages, forKey: .maxPages)

        case .urlQueued(let jobId, let url, let depth, let source):
            try container.encode("url_queued", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(url, forKey: .url)
            try container.encode(depth, forKey: .depth)
            try container.encode(source, forKey: .source)

        case .urlFetching(let jobId, let url):
            try container.encode("url_fetching", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(url, forKey: .url)

        case .urlIndexed(let jobId, let url, let title, let wordCount, let chunksCreated, let durationMs, let discoveredUrls):
            try container.encode("url_indexed", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(url, forKey: .url)
            try container.encodeIfPresent(title, forKey: .title)
            try container.encode(wordCount, forKey: .wordCount)
            try container.encode(chunksCreated, forKey: .chunksCreated)
            try container.encode(durationMs, forKey: .durationMs)
            try container.encode(discoveredUrls, forKey: .discoveredUrls)

        case .urlFailed(let jobId, let url, let error, let durationMs):
            try container.encode("url_failed", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(url, forKey: .url)
            try container.encode(error, forKey: .error)
            try container.encode(durationMs, forKey: .durationMs)

        case .urlSkipped(let jobId, let url, let reason):
            try container.encode("url_skipped", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(url, forKey: .url)
            try container.encode(reason, forKey: .reason)

        case .progress(let jobId, let urlsProcessed, let urlsSucceeded, let urlsFailed, let urlsSkipped, let urlsQueued, let chunksIndexed, let elapsedMs, let rate, let etaSeconds):
            try container.encode("progress", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(urlsProcessed, forKey: .urlsProcessed)
            try container.encode(urlsSucceeded, forKey: .urlsSucceeded)
            try container.encode(urlsFailed, forKey: .urlsFailed)
            try container.encode(urlsSkipped, forKey: .urlsSkipped)
            try container.encode(urlsQueued, forKey: .urlsQueued)
            try container.encode(chunksIndexed, forKey: .chunksIndexed)
            try container.encode(elapsedMs, forKey: .elapsedMs)
            try container.encodeIfPresent(rate, forKey: .rate)
            try container.encodeIfPresent(etaSeconds, forKey: .etaSeconds)

        case .jobCompleted(let jobId, let status, let stats, let error):
            try container.encode("job_completed", forKey: .type)
            try container.encode(jobId, forKey: .jobId)
            try container.encode(status, forKey: .status)
            try container.encodeIfPresent(stats, forKey: .stats)
            try container.encodeIfPresent(error, forKey: .error)

        case .lagged(let missed):
            try container.encode("lagged", forKey: .type)
            try container.encode(missed, forKey: .missed)
        }
    }
}
