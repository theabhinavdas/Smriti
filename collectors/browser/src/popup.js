/**
 * Popup script: loads/saves config, checks daemon health status.
 */

const DEFAULTS = {
  enabled: true,
  daemonUrl: "http://127.0.0.1:9898",
  minDwellSeconds: 5,
  batchIntervalMs: 5000,
  maxQueueSize: 1000,
  blocklist: ["*.bank.com", "mail.google.com", "*.1password.com", "accounts.google.com"],
  extractors: { chatgpt: true, gemini: true, claude: true },
};

async function loadConfig() {
  const stored = await chrome.storage.local.get("smritiConfig");
  return { ...DEFAULTS, ...stored.smritiConfig };
}

async function saveConfig(config) {
  await chrome.storage.local.set({ smritiConfig: config });
}

document.addEventListener("DOMContentLoaded", async () => {
  const config = await loadConfig();

  const $enabled = document.getElementById("enabled");
  const $daemonUrl = document.getElementById("daemonUrl");
  const $minDwell = document.getElementById("minDwell");
  const $extChatgpt = document.getElementById("extChatgpt");
  const $extGemini = document.getElementById("extGemini");
  const $extClaude = document.getElementById("extClaude");
  const $blocklist = document.getElementById("blocklist");
  const $save = document.getElementById("save");
  const $saveStatus = document.getElementById("saveStatus");
  const $statusDot = document.getElementById("statusDot");
  const $statusText = document.getElementById("statusText");

  $enabled.checked = config.enabled;
  $daemonUrl.value = config.daemonUrl;
  $minDwell.value = config.minDwellSeconds;
  $extChatgpt.checked = config.extractors?.chatgpt ?? true;
  $extGemini.checked = config.extractors?.gemini ?? true;
  $extClaude.checked = config.extractors?.claude ?? true;
  $blocklist.value = (config.blocklist || []).join("\n");

  // Check daemon status
  try {
    const resp = await fetch(`${config.daemonUrl}/v1/health`);
    if (resp.ok) {
      $statusDot.className = "dot connected";
      const data = await resp.json();
      $statusText.textContent = `Connected (${Math.round(data.uptime_seconds)}s uptime)`;
    } else {
      $statusDot.className = "dot disconnected";
      $statusText.textContent = `Error: ${resp.status}`;
    }
  } catch {
    $statusDot.className = "dot disconnected";
    $statusText.textContent = "Daemon unreachable";
  }

  $save.addEventListener("click", async () => {
    const updated = {
      enabled: $enabled.checked,
      daemonUrl: $daemonUrl.value.replace(/\/+$/, ""),
      minDwellSeconds: parseInt($minDwell.value, 10) || 5,
      batchIntervalMs: config.batchIntervalMs,
      maxQueueSize: config.maxQueueSize,
      blocklist: $blocklist.value
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      extractors: {
        chatgpt: $extChatgpt.checked,
        gemini: $extGemini.checked,
        claude: $extClaude.checked,
      },
    };

    await saveConfig(updated);
    $saveStatus.textContent = "Saved";
    setTimeout(() => ($saveStatus.textContent = ""), 2000);
  });
});
