/**
 * Google Gemini conversation extractor (gemini.google.com).
 *
 * Parses conversation turns from the Gemini web interface.
 */

(() => {
  "use strict";

  const SELECTORS = {
    chatContainer: "main, .conversation-container",
    userMessage: ".query-text, [data-text-role='user'], .user-query",
    modelMessage: ".response-text, .model-response-text, [data-text-role='model'], .markdown",
    turnContainer: ".conversation-turn, .turn-container, [class*='turn']",
  };

  class GeminiExtractor extends BaseExtractor {
    constructor() {
      super({
        source: "gemini",
        selectors: SELECTORS,
        debounceMs: 2500,
      });
    }

    _extractMessages() {
      const messages = [];

      const userEls = document.querySelectorAll(SELECTORS.userMessage);
      const modelEls = document.querySelectorAll(SELECTORS.modelMessage);

      const turns = Math.max(userEls.length, modelEls.length);
      for (let i = 0; i < turns; i++) {
        if (i < userEls.length) {
          const text = userEls[i].innerText.trim();
          if (text) messages.push({ role: "user", content: text });
        }
        if (i < modelEls.length) {
          const text = modelEls[i].innerText.trim();
          if (text) messages.push({ role: "assistant", content: text });
        }
      }

      return messages;
    }

    _detectModel() {
      const modelBadge = document.querySelector(
        "[class*='model-badge'], [class*='model-selector']"
      );
      if (modelBadge) {
        const text = modelBadge.innerText.trim().toLowerCase();
        if (text.includes("2.0")) return "gemini-2.0";
        if (text.includes("1.5")) return "gemini-1.5";
        if (text.includes("ultra")) return "gemini-ultra";
        if (text.includes("flash")) return "gemini-flash";
      }
      return "gemini";
    }
  }

  const extractor = new GeminiExtractor();
  extractor.startObserving();
})();
