#!/usr/bin/env python3
"""
Aganthos Intake Form Assistant
================================
Helps answer corporate partnership intake forms for Aganthos.

Usage:
  python main.py answer --form "M42"
  python main.py answer --form "Siemens"
  python main.py answer --form "cyber innovation hub"
  python main.py answer --form "custom"          # paste in a form interactively
  python main.py answer --form path/to/form.txt  # read from file
  python main.py list                            # list available forms + leads
  python main.py ask "What is Aganthos' TRL level?"  # quick one-off question
"""

import sys
import os
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import print as rprint

console = Console()

FORMS_DIR        = Path(__file__).parent / "intake forms_instructions"
SUGGESTIONS_DIR  = Path(__file__).parent / "intake forms_claude suggestion"
ADAPTED_DIR      = Path(__file__).parent / "intake form_adapted answers for uploading"

# Known intake form leads (expand as you find more)
INTAKE_FORM_LEADS = [
    {
        "org": "M42 (Abu Dhabi Healthcare)",
        "url": "https://forms.monday.com/forms/8431537cb10cb8862d821694fdb3069a?r=euc1",
        "sector": "Healthcare",
        "why": "Large integrated healthcare system in UAE seeking digital transformation partners. Perfect fit for our hospital QA agent case study.",
        "file": "M42",
    },
    {
        "org": "Siemens",
        "url": "https://www.siemens.com/en-us/company/innovation/startups/collaborate/",
        "sector": "Manufacturing / Industrial",
        "why": "Venture client program — Siemens pays for pilots. Good fit for IT, R&D, and production workflow optimisation.",
        "file": "Siemens",
    },
    {
        "org": "Cyber Innovation Hub (Bundeswehr)",
        "url": "https://www.cyberinnovationhub.de/innovation/dein-cihbw-pitch#c3711",
        "sector": "Defence / Dual-Use",
        "why": "Focused on Software-Defined Defence (AI/RL for military) and UxS. Aganthos RL fits SDD focus area perfectly.",
        "file": "cyber innovation hub",
    },
    {
        "org": "Bayer G4A (Grants4Apps)",
        "url": "https://www.g4a.health/apply",
        "sector": "Healthcare",
        "why": "Bayer's digital health innovation program. Non-dilutive grants + co-creation. Strong fit for clinical AI.",
        "file": None,
    },
    {
        "org": "Bosch Startup Harbour",
        "url": "https://www.bosch-startup.com/en/apply",
        "sector": "Manufacturing / Industrial",
        "why": "Venture client: paid pilots with Bosch. Industrial AI, quality control, production optimisation.",
        "file": None,
    },
    {
        "org": "BMW Startup Garage",
        "url": "https://www.bmwstartupgarage.com/apply",
        "sector": "Manufacturing / Automotive",
        "why": "Paid pilot model — not equity. AI for production, quality, and autonomous systems.",
        "file": None,
    },
    {
        "org": "SAP.iO Foundry",
        "url": "https://sap.io/foundries/",
        "sector": "Enterprise Software",
        "why": "No equity taken. Cohort-based. Strong for enterprise AI workflow automation and ERP integration.",
        "file": None,
    },
    {
        "org": "Munich Re / Startup Cooperation",
        "url": "https://www.munichre.com/en/innovation/startup-cooperation.html",
        "sector": "Insurance / Reinsurance",
        "why": "Risk, InsurTech, industrial data AI. RL for claims, underwriting, and risk modelling.",
        "file": None,
    },
    {
        "org": "Roche Startup Partnering",
        "url": "https://www.roche.com/innovation/partnering/startup-and-scaleup-partnering",
        "sector": "Healthcare / Pharma",
        "why": "Clinical AI, diagnostics, drug discovery. Strong overlap with our medical RL model.",
        "file": None,
    },
    {
        "org": "Novartis Biome",
        "url": "https://www.novartis.com/partnerships/startup-engagement",
        "sector": "Healthcare / Pharma",
        "why": "Health AI, biomarker discovery, clinical data automation. ICML submission adds credibility.",
        "file": None,
    },
    {
        "org": "UnternehmerTUM / Corporate Collaboration",
        "url": "https://www.unternehmertum.de/en/services/collaborate",
        "sector": "Deep Tech / Industrial",
        "why": "TUM network — connects deep tech startups with corporates. Strong fit given founders' TUM background.",
        "file": None,
    },
    {
        "org": "BASF Chemovator",
        "url": "https://www.chemovator.com/apply",
        "sector": "Chemical / Industrial",
        "why": "Industrial process AI, supply chain optimisation, R&D automation.",
        "file": None,
    },
]

# Context about each known organisation (used by the answer generator)
ORG_CONTEXTS = {
    "m42": """
M42 is a leading Abu Dhabi-based integrated healthcare company operating hospitals, clinics, diagnostics and digital health platforms across 24+ countries.
They are actively seeking AI solutions for: AI/digital health, precision medicine, life sciences innovation, and health system optimization.
They specifically want proposals demonstrating measurable value in improving patient outcomes and clinical/operational performance.
Key: emphasise our hospital QA agent case study (FHIR integration, +30% accuracy, data sovereignty, potential 5-15 FTE saved).
They care about: regulatory approvals, differentiation, deployment maturity, and specific impact on clinical workflows.
""",
    "siemens": """
Siemens is a global technology company focused on electrification, automation, and digitalization.
Their startup collaboration focuses on: R&D of Products/Services, Production/Manufacturing, Logistics/Supply Chain, IT.
They offer "Collaborate: Venture Clienting" — meaning they pay for pilots.
Key: position as a solution for Siemens' IT workflows and/or R&D productivity agents.
Emphasise: early market stage (we have paying customers), business model clarity, and specific ROI.
""",
    "cyber innovation hub": """
The Cyber Innovation Hub der Bundeswehr (CIHBw) connects startups with the German Armed Forces (Bundeswehr).
Current focus areas:
1. SOFTWARE DEFINED DEFENCE (SDD): AI/software to increase military capability, modular architecture, agile software development
2. (C)UxS: Unmanned systems (land, air, water) and counter-UxS — software-defined components
They run 12-24 month prototype contracts via below-threshold procurement.
Key: frame Aganthos as enabling Software-Defined Defence through RL-trained agents that adapt to changing operational requirements.
Emphasise: dual-use tech, data sovereignty (on-premise), TRL level, and adaptability to new mission constraints.
Note: Must have EWR presence (we have Swiss GmbH ✓).
Language: German preferred.
""",
    "default": """
This is a corporate innovation program seeking startup partnerships.
Emphasise our most compelling proof points: +30% performance over o4-mini, 60% cost reduction, €28K revenue, NVIDIA Inception, ICML submissions, and our unique RL approach.
"""
}


def get_org_context(org_name: str) -> str:
    key = org_name.lower().strip()
    for k, v in ORG_CONTEXTS.items():
        if k in key or key in k:
            return v
    return ORG_CONTEXTS["default"]


def load_form(form_input: str) -> tuple[str, str]:
    """
    Load form text. Returns (form_text, org_name).
    Tries: intake forms_instructions/ → file path → interactive paste.
    """
    form_path = FORMS_DIR / form_input
    if form_path.exists():
        return form_path.read_text(), form_input

    p = Path(form_input)
    if p.exists():
        return p.read_text(), p.stem

    if form_input.lower() == "custom":
        console.print(Panel(
            "[yellow]Paste your form content below.[/yellow]\n"
            "Enter a blank line followed by [bold]END[/bold] to finish.",
            title="Custom Form Input"
        ))
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        return "\n".join(lines), Prompt.ask("Organisation name")

    raise FileNotFoundError(
        f"Form '{form_input}' not found in {FORMS_DIR}.\n"
        "Run 'python main.py list' to see available forms, or 'python main.py add-form \"OrgName\"' to add one."
    )


def cmd_answer(args):
    """Answer all questions in an intake form."""
    from form_answerer import generate_answers

    try:
        form_text, org_name = load_form(args.form)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    org_context = get_org_context(org_name)

    # Check for org-specific language
    lang = "de" if "cyber innovation hub" in org_name.lower() or args.lang == "de" else "auto"

    console.print(Panel(
        f"[bold cyan]Generating answers for:[/bold cyan] [white]{org_name}[/white]\n"
        f"[dim]Form length: {len(form_text)} chars | Language: {lang}[/dim]",
        title="Aganthos Form Assistant"
    ))

    answers = generate_answers(
        form_text=form_text,
        org_name=org_name,
        org_context=org_context,
        language=lang,
        verbose=True,
    )

    # Display
    console.print("\n")
    console.print(Panel(Markdown(answers), title=f"[bold green]Answers: {org_name}[/bold green]", expand=False))

    # Save to intake forms_claude suggestion/
    SUGGESTIONS_DIR.mkdir(exist_ok=True)
    save_path = SUGGESTIONS_DIR / f"answers_{org_name.replace(' ', '_').lower()}.md"
    if args.save or Confirm.ask(f"\nSave to [cyan]{save_path}[/cyan]?", default=True):
        save_path.write_text(f"# Intake Form Answers: {org_name}\n\n{answers}\n")
        console.print(f"[green]Saved to {save_path}[/green]")
        console.print(f"[dim]Next step: review, then copy to '{ADAPTED_DIR.name}/' before submitting.[/dim]")


def cmd_list(args):
    """List all known intake form leads."""
    console.print(Panel("[bold]Known Intake Form Leads for Aganthos[/bold]", style="cyan"))
    for i, lead in enumerate(INTAKE_FORM_LEADS, 1):
        has_form = "✓ form on file" if lead["file"] else "  no form yet"
        console.print(
            f"[bold cyan]{i:2}. {lead['org']}[/bold cyan] "
            f"[dim]({lead['sector']})[/dim] [{has_form}]\n"
            f"    [dim]{lead['why']}[/dim]\n"
            f"    [blue underline]{lead['url']}[/blue underline]\n"
        )


def cmd_ask(args):
    """Answer a quick one-off question about how Aganthos should respond to something."""
    from form_answerer import answer_single_question
    question = " ".join(args.question)
    org = args.org or "a corporate partner"
    lang = "de" if args.lang == "de" else "en"

    console.print(f"\n[dim]Question:[/dim] {question}\n")
    answer = answer_single_question(
        question=question,
        form_context="",
        org_name=org,
        language=lang,
        char_limit=args.limit or 0,
    )
    console.print(Panel(Markdown(answer), title="Answer"))


def cmd_add_form(args):
    """Add a new intake form to intake forms_instructions/."""
    FORMS_DIR.mkdir(exist_ok=True)
    out_path = FORMS_DIR / args.name
    if out_path.exists() and not Confirm.ask(f"[yellow]{out_path} exists. Overwrite?[/yellow]"):
        return

    console.print(Panel(
        "[yellow]Paste the form content (questions, instructions, options).\n"
        "Include the URL on line 1. Enter END on a blank line to finish.[/yellow]",
        title=f"Adding form: {args.name}"
    ))
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    out_path.write_text("\n".join(lines))
    console.print(f"[green]Saved to {out_path}[/green]")
    console.print(f"[dim]Run: python main.py answer --form \"{args.name}\"[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Aganthos Intake Form Assistant — generate compelling answers to partnership forms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py answer --form "M42"
  python main.py answer --form "Siemens"
  python main.py answer --form "cyber innovation hub"
  python main.py answer --form "custom"
  python main.py list
  python main.py ask "What is our TRL level?" --org "Siemens"
  python main.py ask "Describe unfair advantage" --org "Bundeswehr" --lang de --limit 500
  python main.py add-form "Bosch"
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # answer
    p_answer = subparsers.add_parser("answer", help="Generate answers for an intake form")
    p_answer.add_argument("--form", required=True, help="Form name (e.g. 'M42'), file path, or 'custom'")
    p_answer.add_argument("--lang", default="auto", choices=["auto", "en", "de"], help="Language")
    p_answer.add_argument("--save", action="store_true", help="Auto-save answers without prompting")

    # list
    p_list = subparsers.add_parser("list", help="List all known intake form leads")

    # ask
    p_ask = subparsers.add_parser("ask", help="Answer a single question")
    p_ask.add_argument("question", nargs="+", help="The question to answer")
    p_ask.add_argument("--org", default="", help="Target organisation")
    p_ask.add_argument("--lang", default="en", choices=["en", "de"])
    p_ask.add_argument("--limit", type=int, default=0, help="Character limit for answer")

    # add-form
    p_add = subparsers.add_parser("add-form", help="Add a new intake form")
    p_add.add_argument("name", help="Name for the form file")

    args = parser.parse_args()

    if args.command == "answer":
        cmd_answer(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "add-form":
        cmd_add_form(args)
    else:
        parser.print_help()
        console.print("\n[dim]Quick start:[/dim]")
        console.print("  python main.py list                    # see all leads")
        console.print("  python main.py answer --form M42       # answer M42 form")
        console.print("  python main.py answer --form custom    # paste in any form")


if __name__ == "__main__":
    main()
