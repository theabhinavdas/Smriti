/**
 * ChatGPT conversation extractor (chatgpt.com / chat.openai.com).
 *
 * Parses conversation messages from the DOM using data attributes
 * and class patterns specific to the ChatGPT web interface.
 */

(() => {
  "use strict";

  const SELECTORS = {
    chatContainer: "main",
    messageGroup: "[data-message-author-role]",
    messageContent: ".markdown, .whitespace-pre-wrap",
    modelSelector: "[class*='model']",
    titleElement: "nav a.bg-token-sidebar-surface-secondary, nav .truncate",
  };

  class ChatGPTExtractor extends BaseExtractor {
    constructor() {
      super({
        source: "chatgpt",
        selectors: SELECTORS,
        debounceMs: 2000,
      });
    }

    _extractMessages() {
      const elements = document.querySelectorAll(SELECTORS.messageGroup);
      const messages = [];

      for (const el of elements) {
        const role = el.getAttribute("data-message-author-role");
        if (!role || (role !== "user" && role !== "assistant")) continue;

        const contentEl = el.querySelector(SELECTORS.messageContent);
        if (!contentEl) continue;

        const text = contentEl.innerText.trim();
        if (!text) continue;

        messages.push({ role, content: text });
      }

      return messages;
    }

    _detectModel() {
      const modelEl = document.querySelector(
        "button[class*='group'] span, [data-testid*='model']"
      );
      if (modelEl) {
        const text = modelEl.innerText.trim().toLowerCase();
        if (text.includes("4o")) return "gpt-4o";
        if (text.includes("4")) return "gpt-4";
        if (text.includes("3.5")) return "gpt-3.5-turbo";
        if (text.includes("o1")) return "o1";
        if (text.includes("o3")) return "o3";
      }
      return "chatgpt";
    }

    _getTitle() {
      const titleEl = document.querySelector(SELECTORS.titleElement);
      return titleEl?.innerText?.trim() || document.title;
    }
  }

  const extractor = new ChatGPTExtractor();
  extractor.startObserving();
})();
