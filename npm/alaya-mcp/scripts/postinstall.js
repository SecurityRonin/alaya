"use strict";

// Postinstall fallback: if the platform-specific optionalDependency was not
// installed (e.g. --ignore-optional, unsupported package manager), download
// the binary from the GitHub Release.

const fs = require("fs");
const path = require("path");
const https = require("https");
const { execSync } = require("child_process");

const PLATFORMS = {
  "darwin-arm64": { artifact: "alaya-mcp-darwin-arm64", binary: "alaya-mcp", pkg: "@alaya-mcp/cli-darwin-arm64" },
  "darwin-x64": { artifact: "alaya-mcp-darwin-x64", binary: "alaya-mcp", pkg: "@alaya-mcp/cli-darwin-x64" },
  "linux-x64": { artifact: "alaya-mcp-linux-x64", binary: "alaya-mcp", pkg: "@alaya-mcp/cli-linux-x64" },
  "win32-x64": { artifact: "alaya-mcp-win32-x64", binary: "alaya-mcp.exe", pkg: "@alaya-mcp/cli-win32-x64" },
};

function main() {
  const platformKey = `${process.platform}-${process.arch}`;
  const platform = PLATFORMS[platformKey];
  if (!platform) {
    // Unsupported platform — run.js will report a clear error
    return;
  }

  // Check if optionalDependency already installed
  try {
    require.resolve(`${platform.pkg}/package.json`);
    return; // Binary available via optionalDependencies
  } catch {
    // Not installed — download from GitHub Release
  }

  const pkg = require("../package.json");
  const version = pkg.version;
  const isWindows = process.platform === "win32";
  const ext = isWindows ? "zip" : "tar.gz";
  const url = `https://github.com/SecurityRonin/alaya/releases/download/v${version}/${platform.artifact}.${ext}`;
  const binDir = path.join(__dirname, "..", "bin");
  const dest = path.join(binDir, platform.binary);

  // Skip if already downloaded
  if (fs.existsSync(dest)) {
    return;
  }

  console.log(`alaya-mcp: downloading binary for ${platformKey}...`);

  const tmpFile = path.join(binDir, `download.${ext}`);
  fs.mkdirSync(binDir, { recursive: true });

  try {
    // Use curl/wget for simplicity and redirect handling
    try {
      execSync(`curl -fsSL "${url}" -o "${tmpFile}"`, { stdio: "pipe" });
    } catch {
      execSync(`wget -q "${url}" -O "${tmpFile}"`, { stdio: "pipe" });
    }

    // Extract
    if (isWindows) {
      execSync(`powershell -Command "Expand-Archive -Path '${tmpFile}' -DestinationPath '${binDir}' -Force"`, { stdio: "pipe" });
    } else {
      execSync(`tar xzf "${tmpFile}" -C "${binDir}"`, { stdio: "pipe" });
      fs.chmodSync(dest, 0o755);
    }

    // Cleanup
    fs.unlinkSync(tmpFile);
    console.log(`alaya-mcp: binary installed successfully`);
  } catch (err) {
    console.warn(
      `alaya-mcp: failed to download binary (${err.message})\n` +
        `You can build from source: cargo build --release --features "mcp llm"`
    );
    // Clean up partial downloads
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

main();

// Report anonymous install event (fire-and-forget, respects opt-out)
require("../lib/analytics").reportInstall().catch(() => {});
