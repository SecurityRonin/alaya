"use strict";

const https = require("https");
const http = require("http");
const path = require("path");

// Scarf pixel endpoint for install tracking
const DEFAULT_ENDPOINT =
  "https://static.scarf.sh/a.png?x-pxid=c18dc510-7aec-427a-868b-2753233f9a35";

/**
 * Check if the user has opted out of install analytics.
 *
 * Respects:
 *   SCARF_ANALYTICS=false   (Scarf standard)
 *   DO_NOT_TRACK=1          (W3C standard)
 *   ALAYA_TELEMETRY=false   (project-specific)
 */
function isOptedOut() {
  const env = process.env;
  if (env.SCARF_ANALYTICS === "false") return true;
  if (env.DO_NOT_TRACK === "1") return true;
  if (env.ALAYA_TELEMETRY === "false") return true;
  return false;
}

/**
 * Build a telemetry payload with zero PII.
 * Only sends: package name, version, platform, arch, node version.
 */
function buildPayload() {
  let version = "0.0.0";
  try {
    const pkg = require(path.join(__dirname, "..", "package.json"));
    version = pkg.version || version;
  } catch {
    // package.json not readable — use fallback
  }

  return {
    package: "alaya-mcp",
    version,
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
  };
}

/**
 * Report install event to Scarf. Fire-and-forget with timeout.
 *
 * Returns:
 *   "opted-out" — user opted out
 *   "sent"      — request sent successfully
 *   "error"     — request failed (swallowed)
 *
 * Options:
 *   endpoint — override the Scarf pixel URL
 *   timeout  — HTTP timeout in ms (default 3000)
 */
function reportInstall(options = {}) {
  return new Promise((resolve) => {
    if (isOptedOut()) {
      return resolve("opted-out");
    }

    const payload = buildPayload();
    const endpoint = options.endpoint || DEFAULT_ENDPOINT;
    const timeout = options.timeout || 3000;

    try {
      const url = new URL(endpoint);

      // Append payload as query params (Scarf pixel approach)
      url.searchParams.set("package", payload.package);
      url.searchParams.set("version", payload.version);
      url.searchParams.set("platform", payload.platform);
      url.searchParams.set("arch", payload.arch);
      url.searchParams.set("node", payload.nodeVersion);

      const transport = url.protocol === "https:" ? https : http;

      const req = transport.get(url.toString(), { timeout }, (res) => {
        // Consume response to free socket
        res.resume();
        resolve("sent");
      });

      req.on("error", () => resolve("error"));
      req.on("timeout", () => {
        req.destroy();
        resolve("error");
      });
    } catch {
      resolve("error");
    }
  });
}

module.exports = { isOptedOut, buildPayload, reportInstall };
