/**
 * Claude conversation extractor (claude.ai).
 *
 * Parses conversation turns from the Anthropic Claude web interface.
 */

(() => {
  "use strict";

  const SELECTORS = {
    chatContainer: "[class*='conversation'], main",
    humanMessage: "[data-is-streaming='false'] .font-user-message, [class*='human-turn'] .whitespace-pre-wrap",
    assistantMessage: "[data-is-streaming='false'] .font-claude-message, [class*='assistant-turn'] .markdown",
    turnContainer: "[class*='turn'], [class*='message-row']",
  };

  class ClaudeExtractor extends BaseExtractor {
    constructor() {
      super({
        source: "claude",
        selectors: SELECTORS,
        debounceMs: 2000,
      });
    }

    _extractMessages() {
      const messages = [];

      const humanEls = document.querySelectorAll(SELECTORS.humanMessage);
      const assistantEls = document.querySelectorAll(SELECTORS.assistantMessage);

      const turns = Math.max(humanEls.length, assistantEls.length);
      for (let i = 0; i < turns; i++) {
        if (i < humanEls.length) {
          const text = humanEls[i].innerText.trim();
          if (text) messages.push({ role: "user", content: text });
        }
        if (i < assistantEls.length) {
          const text = assistantEls[i].innerText.trim();
          if (text) messages.push({ role: "assistant", content: text });
        }
      }

      return messages;
    }

    _detectModel() {
      const selector = document.querySelector(
        "[class*='model-selector'], button[class*='model']"
      );
      if (selector) {
        const text = selector.innerText.trim().toLowerCase();
        if (text.includes("opus")) return "claude-opus";
        if (text.includes("sonnet")) return "claude-sonnet";
        if (text.includes("haiku")) return "claude-haiku";
      }
      return "claude";
    }
  }

  const extractor = new ClaudeExtractor();
  extractor.startObserving();
})();
