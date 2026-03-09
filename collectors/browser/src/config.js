/**
 * Configuration management for the Smriti browser collector.
 * Settings are persisted in chrome.storage.local.
 */

const DEFAULTS = {
  enabled: true,
  daemonUrl: "http://127.0.0.1:9898",
  minDwellSeconds: 5,
  batchIntervalMs: 5000,
  maxQueueSize: 1000,
  blocklist: [
    "*.bank.com",
    "mail.google.com",
    "*.1password.com",
    "accounts.google.com",
    "127.0.0.1",
    "localhost",
  ],
  extractors: {
    chatgpt: true,
    gemini: true,
    claude: true,
  },
};

async function loadConfig() {
  const stored = await chrome.storage.local.get("smritiConfig");
  return { ...DEFAULTS, ...stored.smritiConfig };
}

async function saveConfig(config) {
  await chrome.storage.local.set({ smritiConfig: config });
}

function isDomainBlocked(url, blocklist) {
  try {
    const hostname = new URL(url).hostname;
    return blocklist.some((pattern) => {
      if (pattern.startsWith("*.")) {
        const suffix = pattern.slice(1);
        return hostname.endsWith(suffix) || hostname === pattern.slice(2);
      }
      return hostname === pattern;
    });
  } catch {
    return false;
  }
}

// Export for use by other modules (ES module context in service worker)
// and via globalThis for content scripts (non-module context)
if (typeof globalThis !== "undefined") {
  globalThis.SmritiConfig = { DEFAULTS, loadConfig, saveConfig, isDomainBlocked };
}

export { DEFAULTS, loadConfig, saveConfig, isDomainBlocked };
