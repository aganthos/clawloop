# Pitch Deck Workflow

## Quick start (for Robert)

```bash
# 1. Pull latest
git pull

# 2. Start local server from the repo root (required for images and PDF export)
python3 -m http.server 8765

# 3. Open a variant in the browser
# (replace skydeck2 with whichever variant you need)
open http://localhost:8765/pitch/deckv3.html?deck=vc&variant=skydeck2

# 4. Export to PDF
npx decktape generic --key=ArrowDown --max-slides=30 -s 1920x1080 \
  "http://localhost:8765/pitch/deckv3.html?deck=vc&variant=skydeck2" \
  pitch/deckv3_skydeck2.pdf

# 5. Compress PDF (~4 MB)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
  -dNOPAUSE -dQUIET -dBATCH \
  -sOutputFile=pitch/deckv3_skydeck2_small.pdf \
  pitch/deckv3_skydeck2.pdf

# 6. Rebuild standalone HTML (no server needed, shareable file)
python3 - <<'EOF'
import base64, re, os, subprocess, tempfile
from html import unescape
src, dst = "pitch/deckv3.html", "pitch/deckv3_standalone.html"
with open(src, "r", encoding="utf-8") as f: html = f.read()
def to_data_uri(path, ref_dir):
    path = unescape(path)
    abs_path = os.path.normpath(os.path.join(ref_dir, path))
    if not os.path.exists(abs_path): return None
    ext = os.path.splitext(abs_path)[1].lower()
    mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
            ".gif":"image/gif",".svg":"image/svg+xml"}.get(ext,"application/octet-stream")
    if abs_path.endswith("monkey.png"):
        tmp = tempfile.mktemp(suffix=".jpg")
        subprocess.run(["sips","-s","format","jpeg","-s","formatOptions","80",abs_path,"--out",tmp],capture_output=True)
        with open(tmp,"rb") as f2: data=f2.read()
        os.unlink(tmp); mime="image/jpeg"
    else:
        with open(abs_path,"rb") as f2: data=f2.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
def replace_src(m):
    path = m.group(1)
    if path.startswith("data:") or path.startswith("http"): return m.group(0)
    uri = to_data_uri(path, "pitch")
    return f'src="{uri}"' if uri else m.group(0)
html_out = re.sub(r'src="([^"]+)"', replace_src, html)
with open(dst,"w",encoding="utf-8") as f: f.write(html_out)
print(f"Done: {os.path.getsize(dst)/1024/1024:.1f} MB")
EOF
```

**Available variants:** `?deck=vc` · `?deck=vc&variant=skydeck` · `?deck=vc&variant=skydeck2` · `?deck=vc&variant=tenity` · `?deck=vc&variant=siemens`

---

## File structure

- `pitch/deckv3.html` — **single source of truth** for all slide versions
- `pitch/deckv3_standalone.html` — self-contained version (images inlined, no server needed)
- `pitch/deckv3.pdf` — generic VC PDF (no personalised slide)

---

## How personalised variants work

Every slide has `data-deck="vc"`. The two slides before the closing slide are
personalised ask/roadmap slides, each tagged with `data-variant="<name>"`.

Current variants:
| Variant | URL param | Slides shown |
|---|---|---|
| Generic VC | `?deck=vc` | all vc slides, **no** variant slides |
| SkyDeck | `?deck=vc&variant=skydeck` | + SkyDeck ask slide |
| Tenity | `?deck=vc&variant=tenity` | + Tenity partnership slide |
| Siemens | `?deck=vc&variant=siemens` | + Siemens value props + pilot roadmap |

When no params are given, **all** slides are shown (useful for editing/preview).

---

## Adding a new personalised variant

1. Open `pitch/deckv3.html`
2. Find the block just before `<!-- SLIDE 15 — CLOSING -->` (~line 1730)
3. Insert one or two new `<section>` elements using this template:

```html
<!-- ═══ SLIDE — [NAME] (personalised) ═══ -->
<section class="slide bg-deep" data-slide="11" data-deck="vc" data-variant="YOURVARIANT">
    <span class="slide-num">12</span>
    <svg class="logo-corner" viewBox="0 0 900 1400" fill="none">
        <path d="M700 450C700 588.071 588.071 700 450 700C311.929 700 200 588.071 200 450H700ZM450 0C509.095 0 567.611 11.6393 622.208 34.2539C676.804 56.8685 726.412 90.0155 768.198 131.802C809.985 173.588 843.131 223.196 865.746 277.792C888.361 332.389 900 390.905 900 450H700C700 311.929 588.071 200 450 200C311.929 200 200 311.929 200 450H0C0 390.905 11.6394 332.389 34.2539 277.793C56.8685 223.196 90.0154 173.588 131.802 131.802C173.588 90.0155 223.196 56.8685 277.792 34.2539C332.389 11.6393 390.905 0 450 0Z" fill="#E8E8E8"/>
        <path d="M200 950C200 811.929 311.929 700 450 700C588.071 700 700 811.929 700 950H200ZM450 1400C390.905 1400 332.389 1388.36 277.792 1365.75C223.196 1343.13 173.588 1309.98 131.802 1268.2C90.0155 1226.41 56.8685 1176.8 34.2539 1122.21C11.6393 1067.61 0 1009.09 0 950H200C200 1088.07 311.929 1200 450 1200C588.071 1200 700 1088.07 700 950H900C900 1009.09 888.361 1067.61 865.746 1122.21C843.131 1176.8 809.985 1226.41 768.198 1268.2C726.412 1309.98 676.804 1343.13 622.208 1365.75C567.611 1388.36 509.095 1400 450 1400Z" fill="#E8E8E8"/>
    </svg>
    <div class="slide-content" style="gap: var(--content-gap);">
        <h2 class="reveal">What [Name] Means for Aganthos</h2>
        <div class="ask-grid">
            <div class="ask-item reveal-scale">
                <h3 style="color:var(--accent-green);">Point 1</h3>
                <p>Description.</p>
            </div>
            <div class="ask-item reveal-scale">
                <h3 style="color:var(--accent-blue);">Point 2</h3>
                <p>Description.</p>
            </div>
            <div class="ask-item reveal-scale">
                <h3 style="color:var(--accent-warm);">Point 3</h3>
                <p>Description.</p>
            </div>
        </div>
        <div class="highlight-box reveal">
            <p><strong>Target:</strong> Your goal here.</p>
        </div>
    </div>
</section>
```

For a **timeline/roadmap slide** (like Siemens slide 2), use `bg-warm` and replace the ask-grid with:

```html
<div class="reveal" style="display:flex;flex-direction:column;gap:clamp(14px,2vh,28px);width:100%;max-width:min(90vw,700px);margin-top:var(--element-gap);">
    <div style="display:flex;align-items:center;gap:clamp(16px,2vw,32px);">
        <span style="flex-shrink:0;min-width:clamp(90px,9vw,130px);text-align:center;padding:0.4em 1em;border-radius:999px;background:rgba(110,207,128,0.12);border:1px solid rgba(110,207,128,0.4);color:var(--accent-green);font-size:var(--small-size);font-weight:600;">Phase 1</span>
        <p style="font-size:var(--body-size);color:var(--text-primary);">What happens in phase 1.</p>
    </div>
    <div style="display:flex;align-items:center;gap:clamp(16px,2vw,32px);">
        <span style="flex-shrink:0;min-width:clamp(90px,9vw,130px);text-align:center;padding:0.4em 1em;border-radius:999px;background:rgba(126,179,207,0.12);border:1px solid rgba(126,179,207,0.4);color:var(--accent-blue);font-size:var(--small-size);font-weight:600;">Phase 2</span>
        <p style="font-size:var(--body-size);color:var(--text-primary);">What happens in phase 2.</p>
    </div>
    <div style="display:flex;align-items:center;gap:clamp(16px,2vw,32px);">
        <span style="flex-shrink:0;min-width:clamp(90px,9vw,130px);text-align:center;padding:0.4em 1em;border-radius:999px;background:rgba(232,168,124,0.12);border:1px solid rgba(232,168,124,0.4);color:var(--accent-warm);font-size:var(--small-size);font-weight:600;">Phase 3</span>
        <p style="font-size:var(--body-size);color:var(--text-primary);">What happens in phase 3.</p>
    </div>
</div>
```

4. Push to GitHub: `git add pitch/deckv3.html && git commit -m "Add [name] variant" && git push`

---

## Viewing a variant in the browser

Run from the **repo root**. Requires a local server (images use `../ressources/` paths):

```bash
python3 -m http.server 8765
# then open: http://localhost:8765/pitch/deckv3.html?deck=vc&variant=siemens
```

For standalone (no server needed) — rebuild after changes:

```bash
python3 pitch/build_standalone.py   # see script below
open pitch/deckv3_standalone.html
```

---

## Generating a PDF

Requires the local server to be running (see above).

```bash
# Generic VC version (no personalised slide)
npx decktape generic --key=ArrowDown --max-slides=30 -s 1920x1080 \
  "http://localhost:8765/pitch/deckv3.html?deck=vc" \
  pitch/deckv3.pdf

# Specific variant (e.g. Siemens)
npx decktape generic --key=ArrowDown --max-slides=30 -s 1920x1080 \
  "http://localhost:8765/pitch/deckv3.html?deck=vc&variant=siemens" \
  pitch/deckv3_siemens.pdf

# Compressed version (~2-4 MB, email-friendly)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
  -dNOPAUSE -dQUIET -dBATCH \
  -sOutputFile=pitch/deckv3_siemens_small.pdf \
  pitch/deckv3_siemens.pdf
```

`/prepress` = ~3-4 MB, good quality. Use `/printer` for ~3 MB or `/screen` for ~1.5 MB if size matters more than sharpness.

---

## Rebuilding deckv3_standalone.html

Run this Python script from the repo root whenever `deckv3.html` changes:

```bash
python3 - <<'EOF'
import base64, re, os, subprocess, tempfile
from html import unescape

src, dst = "pitch/deckv3.html", "pitch/deckv3_standalone.html"
with open(src, "r", encoding="utf-8") as f: html = f.read()

def to_data_uri(path, ref_dir):
    path = unescape(path)
    abs_path = os.path.normpath(os.path.join(ref_dir, path))
    if not os.path.exists(abs_path): return None
    ext = os.path.splitext(abs_path)[1].lower()
    mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
            ".gif":"image/gif",".svg":"image/svg+xml"}.get(ext,"application/octet-stream")
    if abs_path.endswith("monkey.png"):
        tmp = tempfile.mktemp(suffix=".jpg")
        subprocess.run(["sips","-s","format","jpeg","-s","formatOptions","80",abs_path,"--out",tmp],capture_output=True)
        with open(tmp,"rb") as f2: data=f2.read()
        os.unlink(tmp); mime="image/jpeg"
    else:
        with open(abs_path,"rb") as f2: data=f2.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

def replace_src(m):
    path = m.group(1)
    if path.startswith("data:") or path.startswith("http"): return m.group(0)
    uri = to_data_uri(path, "pitch")
    return f'src="{uri}"' if uri else m.group(0)

html_out = re.sub(r'src="([^"]+)"', replace_src, html)
with open(dst,"w",encoding="utf-8") as f: f.write(html_out)
print(f"Done: {os.path.getsize(dst)/1024/1024:.1f} MB")
EOF
```
