#!/usr/bin/env node
/**
 * Convert an HTML slide deck to PDF (one slide per page, 1920x1080).
 *
 * Uses Puppeteer to screenshot each slide, then combines via pdf-lib.
 *
 * Setup:
 *     npm install puppeteer pdf-lib
 *
 * Usage:
 *     node html2pdf.mjs                                    # main_pitch.html -> main_pitch.pdf
 *     node html2pdf.mjs my_deck.html out.pdf               # custom I/O
 *     node html2pdf.mjs main_pitch.html out.pdf --variant skydeck
 */

import puppeteer from "puppeteer";
import { PDFDocument } from "pdf-lib";
import { readFile, writeFile, mkdir, rm } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const WIDTH = 1920;
const HEIGHT = 1080;
const SNAP_DISABLE_CSS = `
  html, body {
    scroll-snap-type: none !important;
    scroll-behavior: auto !important;
    scroll-padding: 0 !important;
  }
  * {
    scroll-snap-type: none !important;
    scroll-snap-align: none !important;
    scroll-snap-stop: normal !important;
    scroll-margin: 0 !important;
  }
`;

async function htmlToPdf(htmlPath, pdfPath, { deck, variant } = {}) {
  const absPath = resolve(htmlPath);
  let url = `file://${absPath}`;

  const params = [];
  if (deck) params.push(`deck=${deck}`);
  if (variant) params.push(`variant=${variant}`);
  if (params.length) url += "?" + params.join("&");

  console.log(`Opening: ${url}`);
  console.log(`Viewport: ${WIDTH}x${HEIGHT}`);

  const tmpDir = resolve(dirname(absPath), ".slide_screenshots");
  await mkdir(tmpDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      `--window-size=${WIDTH},${HEIGHT}`,
      "--allow-file-access-from-files",
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });

  // Navigate and wait for full load
  await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
  // Wait for fonts + settle time
  await page.evaluate(() => (document.fonts ? document.fonts.ready : Promise.resolve()));
  await new Promise((r) => setTimeout(r, 800));

  // Disable scroll-snap + smooth scrolling so programmatic scroll works
  await page.addStyleTag({ content: SNAP_DISABLE_CSS });
  await page.evaluate(() => {
    document.documentElement.style.scrollSnapType = "none";
    document.body.style.scrollSnapType = "none";
    document.documentElement.style.scrollBehavior = "auto";
    document.body.style.scrollBehavior = "auto";
  });

  const slides = await page.$$("section.slide");
  const total = slides.length;
  console.log(`Found ${total} slides`);

  const screenshotPaths = [];

  // Instead of scrolling, hide all slides and show one at a time
  // This bypasses all scroll-snap issues entirely
  await page.evaluate(() => {
    const all = document.querySelectorAll("section.slide");
    all.forEach((el) => {
      el.style.position = "absolute";
      el.style.top = "0";
      el.style.left = "0";
      el.style.width = "100vw";
      el.style.height = "100vh";
      el.style.display = "none";
    });
  });

  for (let i = 0; i < total; i++) {
    // Show only this slide
    await page.evaluate((index) => {
      const all = document.querySelectorAll("section.slide");
      all.forEach((el, j) => {
        el.style.display = j === index ? "flex" : "none";
        if (j === index) el.classList.add("visible");
      });
    }, i);

    // Wait for images in this slide
    const slide = slides[i];
    await page.evaluate((slideEl) => {
      const imgs = slideEl.querySelectorAll("img");
      return Promise.all(
        Array.from(imgs).map((img) => {
          if (img.complete && img.naturalHeight > 0) return Promise.resolve();
          return new Promise((resolve) => {
            img.addEventListener("load", resolve);
            img.addEventListener("error", resolve);
          });
        })
      );
    }, slide);

    await new Promise((r) => setTimeout(r, 400));

    const path = `${tmpDir}/slide_${String(i).padStart(3, "0")}.png`;
    await page.screenshot({ path });
    screenshotPaths.push(path);
    console.log(`  [${i + 1}/${total}] captured slide ${i}`);
  }

  await browser.close();

  // Combine into PDF using pdf-lib
  console.log(`\nStitching ${screenshotPaths.length} slides into PDF...`);
  const pdfDoc = await PDFDocument.create();

  for (const sp of screenshotPaths) {
    const pngBytes = await readFile(sp);
    const pngImage = await pdfDoc.embedPng(pngBytes);

    // Page size in points (1 point = 1/72 inch)
    // At 72 DPI, 1920px = 1920pt, 1080px = 1080pt
    const page = pdfDoc.addPage([WIDTH, HEIGHT]);
    page.drawImage(pngImage, {
      x: 0,
      y: 0,
      width: WIDTH,
      height: HEIGHT,
    });
  }

  const pdfBytes = await pdfDoc.save();
  await writeFile(pdfPath, pdfBytes);
  console.log(`Done! -> ${pdfPath}`);

  // Cleanup
  for (const sp of screenshotPaths) {
    await rm(sp, { force: true });
  }
  await rm(tmpDir, { recursive: true, force: true });
}

// --- CLI ---
const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith("--"));
const htmlInput = positional[0] || "main_pitch.html";
const pdfOutput = positional[1] || htmlInput.replace(".html", ".pdf");

let deck, variant;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--deck" && args[i + 1]) deck = args[i + 1];
  if (args[i] === "--variant" && args[i + 1]) variant = args[i + 1];
}

htmlToPdf(htmlInput, pdfOutput, { deck, variant });
