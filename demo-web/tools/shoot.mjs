// Headless screenshots of the live demo for the erp-ui self-audit.
// node shoot.mjs <base-url> <out-dir> states|offline
// Needs the `playwright` package on NODE_PATH; run through the job queue.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const [base, out, mode = "states"] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const VIEWS = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true } };
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const browser = await chromium.launch();
async function page(view, scheme = "light") {
  const ctx = await browser.newContext({ viewport: { width: VIEWS[view].width, height: VIEWS[view].height }, deviceScaleFactor: VIEWS[view].deviceScaleFactor || 1, isMobile: !!VIEWS[view].isMobile, hasTouch: !!VIEWS[view].hasTouch, colorScheme: scheme });
  const p = await ctx.newPage();
  p.on("pageerror", (e) => log("PAGE ERROR", e.message));
  await p.goto(base, { waitUntil: "networkidle" });
  await p.waitForSelector("#examples .btn");
  return p;
}
const shot = (p, name, full = false) => p.screenshot({ path: path.join(out, `${name}.png`), fullPage: full }).then(() => log("shot", name));

async function run(p, example, view, tag, captureRunning) {
  await p.click(`#examples .btn:has-text("${example}")`);
  await p.click("#run");
  if (captureRunning) {
    await p.waitForSelector("#status:not([hidden])");
    await p.waitForTimeout(2500);
    await shot(p, `${view}-running-${tag}`);
    // Catch a later stage too, if it shows up before the result.
    try {
      await p.waitForFunction(() => /Untangling|Swapping/.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 600000 });
      await shot(p, `${view}-running-stage`);
    } catch { log("no later stage captured"); }
  }
  await p.waitForSelector("#result:not([hidden]), #status .notice", { timeout: 900000 });
  await p.waitForTimeout(500);
  await shot(p, `${view}-result-${tag}`, true);
}

if (mode === "offline") {
  for (const view of Object.keys(VIEWS)) {
    const p = await page(view);
    await p.waitForSelector("#offline:not([hidden])", { timeout: 30000 });
    await shot(p, `${view}-offline`);
  }
} else {
  const examples = JSON.parse(fs.readFileSync(new URL("../site/public/examples.json", import.meta.url)));
  const [a, b, c] = (process.env.SHOOT_EXAMPLES || examples.map((e) => e.title).join("|")).split("|");
  // Desktop: cold run of A (all three steps change), then B (a step changes nothing).
  // Phone: C runs live (progress), then A and B come back from the server's memory.
  let p = await page("desktop");
  await shot(p, "desktop-empty");
  await run(p, a, "desktop", "all-three", true);
  await run(await page("desktop"), b, "desktop", "one-unchanged", false);
  p = await page("phone");
  await shot(p, "phone-empty");
  await run(p, c, "phone", "third", true);
  await run(await page("phone"), a, "phone", "all-three", false);
  await run(await page("phone"), b, "phone", "one-unchanged", false);
  if (process.env.SHOOT_DARK) {
    const p = await page("desktop", "dark");
    await run(p, a, "desktop", "dark", false);
  }
}
await browser.close();
log("done");
