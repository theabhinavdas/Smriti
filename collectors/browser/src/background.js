/**
 * Background service worker for the Smriti browser collector.
 *
 * Responsibilities:
 * - Track tab activation/deactivation for dwell time
 * - Receive messages from content scripts and extractors
 * - Batch and send events to the daemon API
 * - Handle bookmark events
 */

import { SmritiAPI } from "./api.js";
import { loadConfig, isDomainBlocked } from "./config.js";

const api = new SmritiAPI();
const tabState = new Map(); // tabId -> { url, title, activatedAt }

async function init() {
  const config = await loadConfig();
  api.configure({
    daemonUrl: config.daemonUrl,
    batchIntervalMs: config.batchIntervalMs,
    maxQueueSize: config.maxQueueSize,
  });
}

function makeEvent(source, eventType, rawContent, metadata = {}) {
  return {
    source,
    event_type: eventType,
    timestamp: new Date().toISOString(),
    raw_content: rawContent,
    metadata,
  };
}

// -- Tab tracking for dwell time ------------------------------------------

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  await recordDwell();
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url) {
      tabState.set(tabId, {
        url: tab.url,
        title: tab.title || "",
        activatedAt: Date.now(),
      });
    }
  } catch {
    // Tab may have been closed
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active && tab.url) {
    tabState.set(tabId, {
      url: tab.url,
      title: tab.title || "",
      activatedAt: Date.now(),
    });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  recordDwellForTab(tabId);
  tabState.delete(tabId);
});

async function recordDwell() {
  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (activeTab) {
    recordDwellForTab(activeTab.id);
  }
}

async function recordDwellForTab(tabId) {
  const state = tabState.get(tabId);
  if (!state) return;

  const config = await loadConfig();
  if (!config.enabled) return;

  const dwellSeconds = (Date.now() - state.activatedAt) / 1000;
  if (dwellSeconds < config.minDwellSeconds) return;
  if (isDomainBlocked(state.url, config.blocklist)) return;

  api.enqueue(
    makeEvent("browser", "page_visited", state.title, {
      url: state.url,
      title: state.title,
      dwell_seconds: Math.round(dwellSeconds),
    })
  );
}

// -- Bookmark tracking ----------------------------------------------------

chrome.bookmarks.onCreated.addListener(async (_id, bookmark) => {
  const config = await loadConfig();
  if (!config.enabled) return;
  if (bookmark.url && isDomainBlocked(bookmark.url, config.blocklist)) return;

  api.enqueue(
    makeEvent("browser", "page_bookmarked", bookmark.title || "", {
      url: bookmark.url || "",
      title: bookmark.title || "",
    })
  );
});

// -- Message handling from content scripts --------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse);
  return true; // keep the message channel open for async response
});

async function handleMessage(message, sender) {
  const config = await loadConfig();
  if (!config.enabled) return { ok: false, reason: "disabled" };

  const url = sender.tab?.url || message.url || "";
  if (isDomainBlocked(url, config.blocklist)) return { ok: false, reason: "blocked" };

  switch (message.type) {
    case "TEXT_SELECTED":
      api.enqueue(
        makeEvent("browser", "text_selected", message.text, {
          url,
          title: sender.tab?.title || "",
        })
      );
      break;

    case "SEARCH_QUERY":
      api.enqueue(
        makeEvent("browser", "search", message.query, {
          url,
          engine: message.engine,
        })
      );
      break;

    case "AI_CONVERSATION":
      api.enqueue(
        makeEvent(message.source, "conversation", message.content, {
          url,
          title: message.title || sender.tab?.title || "",
          message_count: message.messageCount || 0,
          model: message.model || "unknown",
        })
      );
      break;

    case "AI_MESSAGE":
      api.enqueue(
        makeEvent(message.source, "ai_message", message.content, {
          url,
          role: message.role,
          model: message.model || "unknown",
        })
      );
      break;

    default:
      return { ok: false, reason: "unknown_type" };
  }

  return { ok: true, queued: api.queueSize };
}

// -- Initialize -----------------------------------------------------------
init();
