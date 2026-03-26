#!/usr/bin/env python3
"""
Convert an HTML slide deck to PDF (one slide per page, 1920x1080).

Uses Playwright (Chromium) to screenshot each slide, then stitches
them into a single PDF via Pillow.

Usage:
    # First time only:
    pip install playwright Pillow
    playwright install chromium

    # Then:
    python html2pdf.py                          # defaults to main_pitch.html -> main_pitch.pdf
    python html2pdf.py my_deck.html out.pdf     # custom input/output
    python html2pdf.py main_pitch.html main_pitch.pdf --variant skydeck   # filter variant
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from PIL import Image


def html_to_pdf(
    html_path: str,
    pdf_path: str,
    width: int = 1920,
    height: int = 1080,
    deck: str | None = None,
    variant: str | None = None,
):
    html_file = Path(html_path).resolve()
    if not html_file.exists():
        print(f"ERROR: {html_file} not found")
        sys.exit(1)

    # Build file:// URL with optional query params for deck/variant filtering
    url = f"file://{html_file}"
    params = []
    if deck:
        params.append(f"deck={deck}")
    if variant:
        params.append(f"variant={variant}")
    if params:
        url += "?" + "&".join(params)

    print(f"Opening: {url}")
    print(f"Viewport: {width}x{height}")

    screenshots: list[Path] = []
    tmp_dir = html_file.parent / ".slide_screenshots"
    tmp_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,  # 1x for exact pixel match; use 2 for retina
        )
        page = context.new_page()

        # Navigate and wait for everything to load
        page.goto(url, wait_until="networkidle")
        # Extra wait for fonts / lazy assets
        page.wait_for_timeout(1500)

        # Get all slide elements
        slides = page.query_selector_all("section.slide")
        total = len(slides)
        print(f"Found {total} slides")

        for i, slide in enumerate(slides):
            # Scroll slide into view
            slide.scroll_into_view_if_needed()
            # Mark it visible (triggers your CSS animations)
            slide.evaluate("el => el.classList.add('visible')")
            # Wait for any images inside this slide to fully load
            page.evaluate("""(slideEl) => {
                const imgs = slideEl.querySelectorAll('img');
                return Promise.all(Array.from(imgs).map(img => {
                    if (img.complete) return Promise.resolve();
                    return new Promise((resolve, reject) => {
                        img.addEventListener('load', resolve);
                        img.addEventListener('error', resolve);  // don't block on broken images
                    });
                }));
            }""", slide)
            # Small extra settle time for CSS transitions
            page.wait_for_timeout(300)

            # Screenshot just this slide element (bounding box)
            screenshot_path = tmp_dir / f"slide_{i:03d}.png"
            slide.screenshot(path=str(screenshot_path))
            screenshots.append(screenshot_path)
            print(f"  [{i+1}/{total}] captured slide {i}")

        browser.close()

    # Stitch into PDF
    print(f"\nStitching {len(screenshots)} slides into PDF...")
    images = []
    for sp in screenshots:
        img = Image.open(sp).convert("RGB")
        # Ensure exact target size (in case of sub-pixel rounding)
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        images.append(img)

    if not images:
        print("ERROR: No slides captured!")
        sys.exit(1)

    # PDF at 72 DPI: 1920x1080 px -> 26.67 x 15 inches (landscape)
    # For standard 72 DPI, just save directly -- each image = one page
    images[0].save(
        pdf_path,
        "PDF",
        resolution=72.0,
        save_all=True,
        append_images=images[1:],
    )
    print(f"Done! -> {pdf_path}")

    # Cleanup temp screenshots
    for sp in screenshots:
        sp.unlink(missing_ok=True)
    tmp_dir.rmdir()


if __name__ == "__main__":
    html_input = sys.argv[1] if len(sys.argv) > 1 else "main_pitch.html"
    pdf_output = sys.argv[2] if len(sys.argv) > 2 else html_input.replace(".html", ".pdf")

    # Parse optional --deck and --variant flags
    deck = None
    variant = None
    for i, arg in enumerate(sys.argv):
        if arg == "--deck" and i + 1 < len(sys.argv):
            deck = sys.argv[i + 1]
        if arg == "--variant" and i + 1 < len(sys.argv):
            variant = sys.argv[i + 1]

    html_to_pdf(html_input, pdf_output, deck=deck, variant=variant)
