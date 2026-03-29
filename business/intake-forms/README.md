# Aganthos Intake Form Tool

AI-assisted pipeline for finding, researching, and answering corporate partnership intake forms.

---

## Folder Structure

```
intake_forms/
│
├── intake forms_instructions/          # Raw form questions copied from the web
│   ├── M42
│   ├── Siemens
│   └── cyber innovation hub
│
├── intake forms_claude suggestion/     # Claude-generated draft answers (DO NOT submit directly)
│   ├── answers_m42.md
│   ├── answers_siemens.md
│   └── answers_cyber_innovation_hub.md
│
├── intake form_adapted answers for uploading/  # Human-reviewed, submission-ready answers
│   └── Siemens
│
├── aganthos_profile.py     # Aganthos knowledge base (value props, case studies, team, traction)
├── form_answerer.py        # Claude-powered answer generation engine
├── main.py                 # CLI tool
├── .env.example            # API key template (copy to .env and fill in)
└── README.md               # This file
```

---

## Workflow

```
1. FIND       Find a relevant intake form / collaboration programme online
      ↓
2. SAVE       Copy all questions + instructions into intake forms_instructions/<OrgName>
              (include the URL on line 1)
      ↓
3. GENERATE   Run the CLI to generate a Claude draft:
              python main.py answer --form "<OrgName>"
      ↓
4. REVIEW     Draft is saved to intake forms_claude suggestion/answers_<orgname>.md
              Read critically — Claude knows Aganthos well but you know the relationship
      ↓
5. ADAPT      Copy to intake form_adapted answers for uploading/<OrgName>
              Edit for tone, accuracy, and any personal context Claude doesn't have
      ↓
6. SUBMIT     Paste adapted answers into the live form and upload pitch deck
```

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
# Install dependencies
pip install anthropic rich python-dotenv

# Set up API key (only needed if running outside Claude Code)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**If using Claude Code** (recommended): no API key needed. Run directly from a terminal outside the Claude Code session.

---

## CLI Commands

```bash
# List all known intake form leads (12 organisations with URLs)
python main.py list

# Generate draft answers for a saved form
python main.py answer --form "M42"
python main.py answer --form "Siemens"
python main.py answer --form "cyber innovation hub"

# Paste in any new form interactively
python main.py answer --form "custom"

# Read a form from a file path
python main.py answer --form "path/to/form.txt"

# Answer a single question (useful for refining specific answers)
python main.py ask "What is our TRL level?" --org "Bosch"
python main.py ask "Describe your unfair advantage" --org "Bundeswehr" --lang de

# Add a new intake form to the instructions folder
python main.py add-form "Bosch"
```

---

## Adding a New Organisation

1. Find the intake form URL
2. Run `python main.py add-form "OrgName"` and paste the form content
   — OR — manually create `intake forms_instructions/OrgName` with the URL on line 1 and questions below
3. Add the org to the `INTAKE_FORM_LEADS` list in `main.py` with URL, sector, and a short note on relevance
4. Add an org-specific context block to `ORG_CONTEXTS` in `main.py` (what does this org care about? what language should we speak?)
5. Run `python main.py answer --form "OrgName"` to generate the draft

---

## Updating the Aganthos Profile

All company knowledge lives in `aganthos_profile.py`. Update it when:
- New traction milestones are hit (revenue, partnerships, publications)
- A new case study is available
- The pitch or positioning evolves

This is the single source of truth that feeds all answer generation.

---

## Known Intake Form Leads

Run `python main.py list` to see the full list with URLs. Current pipeline:

| Organisation | Sector | Form on file |
|---|---|---|
| M42 | Healthcare (UAE) | ✓ |
| Siemens | Manufacturing / Venture Client | ✓ |
| Cyber Innovation Hub (Bundeswehr) | Defence / Dual-Use | ✓ |
| Bayer G4A | Healthcare | — |
| Bosch Startup Harbour | Manufacturing | — |
| BMW Startup Garage | Automotive | — |
| SAP.iO Foundry | Enterprise Software | — |
| Munich Re | Insurance / Reinsurance | — |
| Roche | Healthcare / Pharma | — |
| Novartis Biome | Healthcare / Pharma | — |
| UnternehmerTUM | Deep Tech | — |
| BASF Chemovator | Chemical / Industrial | — |

---

## For Co-founders / Collaborators

- **Robert**: The `aganthos_profile.py` is the main file to keep updated as our technical story evolves. If a new benchmark or paper result comes in, update it there and all future form answers will reflect it automatically. The slides (`Aganthos Slides 110325.pdf`) and past application answers (`Past answer applications Aganthos.docx`) are also in the repo for context.
- **Tobias**: The workflow above is the process. Step 4 (review) is the most important — Claude drafts are strong starting points but always need a human pass before submission.
- Questions / issues → open a GitHub issue or ping on Slack.
