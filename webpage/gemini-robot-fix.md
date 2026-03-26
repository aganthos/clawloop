# Gemini Task: Fix Robot + Globe SVG on Aganthos Webpage

## What needs fixing

The robot SVG in `webpage/index.html` (in the "What is an AI Agent?" section, around line 214-271) looks off on the light background. It was adapted from the pitch deck (`pitch/deckv3.html`, lines 1170-1338) but the adaptation is rough.

## Reference: the good version

The pitch deck at `pitch/deckv3.html` has the correct robot+globe+arrows SVG in slide 7 (around lines 1170-1338). That version:
- Uses a cute robot with rounded head, antenna with glowing tip, expressive eyes with highlights, cheek blushes, smile, ear details, body with circle accent, arms with ball joints, feet
- Uses a realistic globe with blue ocean (#5C85A8), green land masses (#7BA656), latitude/longitude lines
- Has curved arrows between them (action top, reward bottom)
- Has an annotation below the agent showing: Agent = Model + Harness, Harness = Prompt + Tools + Memory (with mini icons for each)
- Dark background theme (#0a0202)

## What the website version needs

Adapt the pitch deck's robot+globe SVG for the website's **light background** (#FAF9F0):
- Stroke colors: `#d8d4d0` (pitch deck robot outlines) → something visible on light bg like `#4A4A4A` or `#6B6B6B`
- Accent color: `#e8a87c` (pitch warm) → `#780000` (Aganthos brand red)
- Globe: adapt blue/green globe for light bg (the globe looks great in the pitch deck, keep it similar but ensure contrast)
- Text labels: use `#131314` for "YOUR AI AGENT" / "YOUR WORKFLOWS"
- Arrow labels: "automates tasks" (top) and "learns from outcomes" (bottom)
- Keep the SVG viewBox at `0 0 900 400` and class `w-full max-w-[700px]`

The **annotation below** (Agent = Model + Harness) is done in HTML, not SVG, so don't touch that. Just fix the robot, globe, and arrows SVG.

## Key files

- **Edit**: `webpage/index.html` — the inline SVG between `<!-- Robot + Globe SVG -->` and the closing `</svg>` (around line 215-271)
- **Reference**: `pitch/deckv3.html` lines 1170-1338 — the good robot+globe SVG to adapt from
- **Also reference**: `ressources/agent_environment_ppt.svg` — a simpler version of the same diagram

## Brand colors

- Background: `#FAF9F0`
- Brand red: `#780000`
- Primary text: `#131314`
- Dark gray: `#4A4A4A`
- Medium gray: `#6B6B6B`

## Do NOT change

- The HTML structure around the SVG
- The Agent = Model + Harness equation section below the SVG
- Any other part of index.html
- Just replace the SVG content between the existing `<svg>` and `</svg>` tags

## Verify by opening

Open `webpage/index.html` in Chrome to verify the robot looks good on the light background.
