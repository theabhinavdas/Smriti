/**
 * HTTP client for the Smriti daemon API.
 * Handles batching, offline buffering, and retry logic.
 */

class SmritiAPI {
  constructor(daemonUrl = "http://127.0.0.1:9898") {
    this._daemonUrl = daemonUrl;
    this._queue = [];
    this._maxQueueSize = 1000;
    this._batchIntervalMs = 5000;
    this._flushThreshold = 10;
    this._timer = null;
    this._flushing = false;
  }

  configure({ daemonUrl, batchIntervalMs, maxQueueSize }) {
    if (daemonUrl) this._daemonUrl = daemonUrl;
    if (batchIntervalMs) this._batchIntervalMs = batchIntervalMs;
    if (maxQueueSize) this._maxQueueSize = maxQueueSize;
    this._startTimer();
  }

  enqueue(event) {
    if (this._queue.length >= this._maxQueueSize) {
      this._queue.shift();
    }
    this._queue.push(event);

    if (this._queue.length >= this._flushThreshold) {
      this.flush();
    }
  }

  async flush() {
    if (this._flushing || this._queue.length === 0) return;
    this._flushing = true;

    const batch = this._queue.splice(0, 500);
    try {
      const resp = await fetch(`${this._daemonUrl}/v1/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: batch }),
      });

      if (!resp.ok) {
        this._queue.unshift(...batch);
        console.warn(`[smriti] API returned ${resp.status}, re-queued ${batch.length} events`);
      }
    } catch (err) {
      this._queue.unshift(...batch);
      console.warn(`[smriti] Daemon unreachable, buffered ${this._queue.length} events`);
    } finally {
      this._flushing = false;
    }
  }

  _startTimer() {
    if (this._timer) clearInterval(this._timer);
    this._timer = setInterval(() => this.flush(), this._batchIntervalMs);
  }

  get queueSize() {
    return this._queue.length;
  }
}

if (typeof globalThis !== "undefined") {
  globalThis.SmritiAPI = SmritiAPI;
}

export { SmritiAPI };
