/**
 * Generic content script injected on all sites.
 *
 * Captures:
 * - Text selections (mouseup with non-empty selection)
 * - Search queries from known search engines
 */

(() => {
  "use strict";

  const MIN_SELECTION_LENGTH = 10;
  let lastSelection = "";

  // -- Text selection tracking --------------------------------------------

  document.addEventListener("mouseup", () => {
    const sel = window.getSelection();
    if (!sel) return;
    const text = sel.toString().trim();
    if (text.length < MIN_SELECTION_LENGTH) return;
    if (text === lastSelection) return;
    lastSelection = text;

    chrome.runtime.sendMessage({
      type: "TEXT_SELECTED",
      text: text.slice(0, 5000),
      url: location.href,
    });
  });

  // -- Search query detection ---------------------------------------------

  const SEARCH_ENGINES = [
    {
      host: "www.google.com",
      param: "q",
      engine: "google",
    },
    {
      host: "duckduckgo.com",
      param: "q",
      engine: "duckduckgo",
    },
    {
      host: "www.bing.com",
      param: "q",
      engine: "bing",
    },
    {
      host: "github.com",
      pathPattern: /\/search/,
      param: "q",
      engine: "github",
    },
    {
      host: "stackoverflow.com",
      pathPattern: /\/search/,
      param: "q",
      engine: "stackoverflow",
    },
  ];

  function detectSearchQuery() {
    const url = new URL(location.href);
    for (const se of SEARCH_ENGINES) {
      if (url.hostname !== se.host) continue;
      if (se.pathPattern && !se.pathPattern.test(url.pathname)) continue;
      const query = url.searchParams.get(se.param);
      if (query && query.trim()) {
        chrome.runtime.sendMessage({
          type: "SEARCH_QUERY",
          query: query.trim(),
          engine: se.engine,
          url: location.href,
        });
        return;
      }
    }
  }

  detectSearchQuery();
})();
