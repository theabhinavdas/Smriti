/**
 * Base extractor class shared by all AI chat site extractors.
 *
 * Provides common logic for:
 * - MutationObserver setup on a chat container
 * - Message deduplication within a session (content hash)
 * - Sending conversation/message events to the background script
 */

class BaseExtractor {
  constructor(siteConfig) {
    this.source = siteConfig.source;
    this.selectors = siteConfig.selectors;
    this._seenHashes = new Set();
    this._observer = null;
    this._extractionTimer = null;
    this._debounceMs = siteConfig.debounceMs || 2000;
  }

  async _hashText(text) {
    const data = new TextEncoder().encode(text);
    const buf = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
      .slice(0, 16);
  }

  startObserving() {
    const target = document.querySelector(this.selectors.chatContainer);
    if (!target) {
      setTimeout(() => this.startObserving(), 2000);
      return;
    }

    this._observer = new MutationObserver(() => {
      clearTimeout(this._extractionTimer);
      this._extractionTimer = setTimeout(() => this._onMutation(), this._debounceMs);
    });

    this._observer.observe(target, { childList: true, subtree: true });

    // Initial extraction for already-loaded conversations
    setTimeout(() => this._extractFullConversation(), 1000);
  }

  stopObserving() {
    if (this._observer) {
      this._observer.disconnect();
      this._observer = null;
    }
  }

  async _onMutation() {
    const messages = this._extractMessages();
    if (messages.length === 0) return;

    const lastMsg = messages[messages.length - 1];
    const hash = await this._hashText(lastMsg.content);
    if (this._seenHashes.has(hash)) return;
    this._seenHashes.add(hash);

    chrome.runtime.sendMessage({
      type: "AI_MESSAGE",
      source: this.source,
      content: lastMsg.content.slice(0, 10000),
      role: lastMsg.role,
      model: this._detectModel(),
    });
  }

  async _extractFullConversation() {
    const messages = this._extractMessages();
    if (messages.length === 0) return;

    const fullText = messages
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n\n");

    const hash = await this._hashText(fullText);
    if (this._seenHashes.has(hash)) return;
    this._seenHashes.add(hash);

    for (const msg of messages) {
      const msgHash = await this._hashText(msg.content);
      this._seenHashes.add(msgHash);
    }

    chrome.runtime.sendMessage({
      type: "AI_CONVERSATION",
      source: this.source,
      content: fullText.slice(0, 50000),
      title: this._getTitle(),
      messageCount: messages.length,
      model: this._detectModel(),
    });
  }

  /** Override in subclass: extract messages from the DOM. */
  _extractMessages() {
    return [];
  }

  /** Override in subclass: detect which model is being used. */
  _detectModel() {
    return "unknown";
  }

  /** Override in subclass: get conversation title. */
  _getTitle() {
    return document.title;
  }
}

// Expose globally for content scripts (not ES modules)
globalThis.BaseExtractor = BaseExtractor;
