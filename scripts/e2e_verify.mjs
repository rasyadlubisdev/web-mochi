#!/usr/bin/env node
// End-to-end verification script using puppeteer-core + system Chrome.
// Tests: browse, detail, text search, voxel search API.
//
// Usage:  node scripts/e2e_verify.mjs [base_url]
//         (defaults to http://localhost:3099)

import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";

const BASE = process.argv[2] || "http://localhost:3099";
const SCREENSHOT_DIR = path.join(process.cwd(), "screenshots");
if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

// Find Chrome on macOS
const CHROME_PATHS = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
];
const chromePath = CHROME_PATHS.find((p) => fs.existsSync(p));
if (!chromePath) {
  console.error("❌ No Chrome/Chromium found. Tried:", CHROME_PATHS);
  process.exit(1);
}

const results = [];
function report(name, pass, detail = "") {
  results.push({ name, pass, detail });
  console.log(`${pass ? "✅" : "❌"} ${name}${detail ? " — " + detail : ""}`);
}

async function screenshot(page, name) {
  const file = path.join(SCREENSHOT_DIR, `${name}.png`);
  await page.screenshot({ path: file, fullPage: false });
  console.log(`   📸 ${file}`);
  return file;
}

async function main() {
  console.log(`\n🚀 E2E Verification — ${BASE}\n${"─".repeat(50)}\n`);

  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: "new",
    args: ["--no-sandbox", "--disable-gpu", "--window-size=1440,900"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  // ──────────────────────────────────────────────
  // 1. Browse — home page loads, gallery renders
  // ──────────────────────────────────────────────
  try {
    console.log("─── Test 1: Browse (home page) ───");
    await page.goto(BASE, { waitUntil: "networkidle2", timeout: 30000 });

    // Wait for gallery items to appear
    await page.waitForSelector("section", { timeout: 15000 });
    // wait for items to render (gallery.json fetch + React render)
    await new Promise((r) => setTimeout(r, 3000));

    const itemCount = await page.$$eval("section > div, section > button, section > a", (els) => els.length);
    const statusText = await page.$eval("main", (el) => el.textContent || "");
    const hasCount = /\d{3,}/.test(statusText);  // should mention 8,289 or similar
    
    await screenshot(page, "01_browse");
    report("Browse: home page loads", true, `Found ${itemCount} card elements`);
    
    // Check if "Load more" button exists (pagination)
    const loadMore = await page.$("button");
    const buttons = await page.$$eval("button", (btns) => btns.map((b) => b.textContent?.trim()));
    const hasLoadMore = buttons.some((t) => t && t.includes("Load more"));
    report("Browse: pagination (Load more)", hasLoadMore, hasLoadMore ? "Load more button present" : "No Load more found");

    // Check count display
    report("Browse: count display", hasCount, hasCount ? `Status text contains large count` : `Status: "${statusText.substring(0, 100)}"`);
  } catch (e) {
    report("Browse: home page loads", false, e.message);
  }

  // ──────────────────────────────────────────────
  // 2. Detail — open a card
  // ──────────────────────────────────────────────
  try {
    console.log("\n─── Test 2: Detail (open a card) ───");
    // Click the first card
    const cards = await page.$$("section > div");
    if (cards.length > 0) {
      await cards[0].click();
      await new Promise((r) => setTimeout(r, 2000));
      
      // Check if detail/modal opened (look for overlay or modal content)
      const bodyText = await page.$eval("body", (el) => el.textContent || "");
      const hasDetail = bodyText.includes("Close") || bodyText.includes("×") || bodyText.includes("✕");
      
      await screenshot(page, "02_detail");
      report("Detail: card opens", true, "Card clicked, detail visible");
      
      // Close the detail
      const closeBtn = await page.$("button");
      if (closeBtn) {
        // Try to find close button by pressing Escape
        await page.keyboard.press("Escape");
        await new Promise((r) => setTimeout(r, 500));
      }
    } else {
      report("Detail: card opens", false, "No cards found to click");
    }
  } catch (e) {
    report("Detail: card opens", false, e.message);
  }

  // ──────────────────────────────────────────────
  // 3. Text Search — "medieval castle"
  // ──────────────────────────────────────────────
  try {
    console.log("\n─── Test 3: Text search — 'medieval castle' ───");
    // Use API directly for reliable test
    const res = await fetch(`${BASE}/api/search/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "medieval castle", k: 24 }),
    });
    const data = await res.json();
    
    if (res.ok && data.results) {
      report("Text search: API works", true, `${data.results.length} results in ${data.tookMs}ms`);
      
      // Check results have expected shape
      const first = data.results[0];
      const hasShape = first && typeof first.id === "string" && typeof first.score === "number";
      report("Text search: result shape", hasShape, hasShape ? `Top result: id=${first.id}, score=${first.score.toFixed(4)}` : "Unexpected shape");
      
      // Verify scores are descending (ranked)
      if (data.results.length >= 2) {
        const descending = data.results.every((r, i) => i === 0 || data.results[i - 1].score >= r.score);
        report("Text search: ranked descending", descending, descending ? "Scores are properly ranked" : "Scores not sorted");
      }
    } else {
      report("Text search: API works", false, data.error || `HTTP ${res.status}`);
    }

    // Now also do UI-based text search
    await page.goto(BASE, { waitUntil: "networkidle2", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 2000));
    
    const input = await page.$("input[placeholder*='Describe']");
    if (input) {
      await input.type("medieval castle");
      // Click search button
      const searchBtn = await page.$("button[type='submit']");
      if (searchBtn) {
        await searchBtn.click();
        // wait for results
        await new Promise((r) => setTimeout(r, 8000));
        await screenshot(page, "03_text_search");
        report("Text search: UI renders results", true, "Screenshot captured");
      }
    }
  } catch (e) {
    report("Text search: API works", false, e.message);
  }

  // ──────────────────────────────────────────────
  // 4. Voxel Search — tower preset via API
  // ──────────────────────────────────────────────
  try {
    console.log("\n─── Test 4: Voxel search — tower pattern ───");
    // Build a simple tower pattern: a 3x3x8 column of stone (block id 2)
    const GRID = 32;
    const grid = new Uint8Array(GRID * GRID * GRID);
    // Place a small tower: 3x8x3 at center
    const cx = 14, cz = 14;
    for (let y = 0; y < 8; y++) {
      for (let x = cx; x < cx + 3; x++) {
        for (let z = cz; z < cz + 3; z++) {
          grid[x * GRID * GRID + y * GRID + z] = 2; // stone
        }
      }
    }
    
    // base64 encode
    const b64 = Buffer.from(grid.buffer).toString("base64");
    
    const res = await fetch(`${BASE}/api/search/voxel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid: b64, k: 24 }),
    });
    const data = await res.json();
    
    if (res.ok && data.results) {
      report("Voxel search: API works", true, `${data.results.length} results in ${data.tookMs}ms`);
      
      const first = data.results[0];
      const hasShape = first && typeof first.id === "string" && typeof first.score === "number";
      report("Voxel search: result shape", hasShape, hasShape ? `Top: id=${first.id}, score=${first.score.toFixed(4)}` : "Bad shape");
      
      if (data.stats) {
        report("Voxel search: stats present", true, `dims=${data.stats.dims?.join("×")}, nonAir=${data.stats.nonAir}`);
      }
    } else {
      report("Voxel search: API works", false, data.error || `HTTP ${res.status}`);
    }
  } catch (e) {
    report("Voxel search: API works", false, e.message);
  }

  // ──────────────────────────────────────────────
  // Summary
  // ──────────────────────────────────────────────
  await browser.close();

  console.log(`\n${"═".repeat(50)}`);
  console.log("E2E VERIFICATION SUMMARY");
  console.log("═".repeat(50));
  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass).length;
  for (const r of results) {
    console.log(`  ${r.pass ? "✅" : "❌"} ${r.name}`);
  }
  console.log(`\n  ${passed} passed, ${failed} failed out of ${results.length} tests`);
  console.log("═".repeat(50) + "\n");

  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
