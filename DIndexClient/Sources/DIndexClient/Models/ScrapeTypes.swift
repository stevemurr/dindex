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
