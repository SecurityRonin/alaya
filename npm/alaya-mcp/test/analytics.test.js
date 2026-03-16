"use strict";

const { describe, it, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert/strict");

const analytics = require("../lib/analytics");

describe("isOptedOut", () => {
  let origEnv;

  beforeEach(() => {
    origEnv = { ...process.env };
  });

  afterEach(() => {
    process.env = origEnv;
  });

  it("returns false by default", () => {
    delete process.env.SCARF_ANALYTICS;
    delete process.env.DO_NOT_TRACK;
    delete process.env.ALAYA_TELEMETRY;
    assert.equal(analytics.isOptedOut(), false);
  });

  it("returns true when SCARF_ANALYTICS=false", () => {
    process.env.SCARF_ANALYTICS = "false";
    assert.equal(analytics.isOptedOut(), true);
  });

  it("returns true when DO_NOT_TRACK=1", () => {
    process.env.DO_NOT_TRACK = "1";
    assert.equal(analytics.isOptedOut(), true);
  });

  it("returns true when ALAYA_TELEMETRY=false", () => {
    process.env.ALAYA_TELEMETRY = "false";
    assert.equal(analytics.isOptedOut(), true);
  });

  it("returns false when SCARF_ANALYTICS=true", () => {
    process.env.SCARF_ANALYTICS = "true";
    assert.equal(analytics.isOptedOut(), false);
  });
});

describe("buildPayload", () => {
  it("includes required fields", () => {
    const payload = analytics.buildPayload();
    assert.equal(payload.package, "alaya-mcp");
    assert.equal(typeof payload.version, "string");
    assert.equal(typeof payload.platform, "string");
    assert.equal(typeof payload.arch, "string");
    assert.equal(typeof payload.nodeVersion, "string");
    assert.ok(payload.version.length > 0);
  });

  it("platform matches process.platform", () => {
    const payload = analytics.buildPayload();
    assert.equal(payload.platform, process.platform);
  });

  it("arch matches process.arch", () => {
    const payload = analytics.buildPayload();
    assert.equal(payload.arch, process.arch);
  });

  it("does not include PII", () => {
    const payload = analytics.buildPayload();
    const json = JSON.stringify(payload);
    // Should not contain username, hostname, or paths
    assert.equal(json.includes(require("os").userInfo().username), false);
    assert.equal(json.includes(require("os").hostname()), false);
  });
});

describe("reportInstall", () => {
  it("resolves without error when opted out", async () => {
    const origEnv = { ...process.env };
    process.env.SCARF_ANALYTICS = "false";
    try {
      const result = await analytics.reportInstall();
      assert.equal(result, "opted-out");
    } finally {
      process.env = origEnv;
    }
  });

  it("resolves without error when endpoint unreachable", async () => {
    // Pass a bogus endpoint to simulate failure
    const result = await analytics.reportInstall({
      endpoint: "http://localhost:1/nonexistent",
      timeout: 500,
    });
    assert.equal(result, "error");
  });

  it("never rejects (swallows all errors)", async () => {
    // Even with completely invalid input, should resolve
    const result = await analytics.reportInstall({
      endpoint: "not-a-url",
      timeout: 500,
    });
    assert.equal(result, "error");
  });
});
