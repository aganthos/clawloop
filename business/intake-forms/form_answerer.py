"""
Intake form answer generator for Aganthos.
Uses the `claude` CLI (no separate API key needed — reuses your Claude Code auth).
Falls back to the Anthropic SDK if ANTHROPIC_API_KEY is set in environment or .env.
"""

import os
import re
import subprocess
import shutil
from pathlib import Path

# Load .env file if present (optional SDK fallback)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from aganthos_profile import COMPANY_PROFILE, SECTOR_PITCHES


def _call_claude(system: str, user: str, max_tokens: int = 4096) -> str:
    """
    Call Claude via the CLI (preferred — no API key needed) or SDK (fallback).
    """
    # Try claude CLI first
    claude_bin = shutil.which("claude")
    if claude_bin and not os.environ.get("CLAUDECODE"):
        full_prompt = f"{system}\n\n---\n\n{user}"
        result = subprocess.run(
            [claude_bin, "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        # If claude CLI fails, fall through to SDK

    # Fallback: Anthropic SDK (needs ANTHROPIC_API_KEY)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "\n\nNo authentication available.\n"
            "Run this tool from a regular terminal (not inside Claude Code):\n"
            "  cd ~/PycharmProjects/intake_forms\n"
            "  python main.py answer --form M42\n\n"
            "Or set ANTHROPIC_API_KEY in a .env file (see .env.example)."
        )

    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def detect_sector(org_name: str, form_text: str) -> str:
    """Guess the most relevant sector pitch from context."""
    text = (org_name + " " + form_text).lower()
    if any(w in text for w in ["health", "hospital", "clinical", "patient", "medical", "pharma", "m42", "roche", "bayer", "novartis"]):
        return SECTOR_PITCHES["healthcare"]
    if any(w in text for w in ["defense", "defence", "military", "bundeswehr", "security", "nato", "armed"]):
        return SECTOR_PITCHES["defense"]
    if any(w in text for w in ["bank", "finance", "insurance", "allianz", "munich re", "financial"]):
        return SECTOR_PITCHES["finance"]
    if any(w in text for w in ["manufactur", "siemens", "bosch", "bmw", "industrial", "production", "supply chain"]):
        return SECTOR_PITCHES["manufacturing"]
    return ""


def extract_questions(form_text: str) -> list[dict]:
    """
    Parse form text into a list of questions.
    Handles numbered questions and free-form text.
    Returns list of dicts: {number, question, context, options}
    """
    lines = form_text.strip().split("\n")
    questions = []
    current_q = None
    context_lines = []

    # Pattern: lines starting with a number followed by . or )
    q_pattern = re.compile(r"^(\d+)[.)]\s+(.+)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = q_pattern.match(line)
        if m:
            # Save previous question
            if current_q:
                current_q["context"] = "\n".join(context_lines).strip()
                questions.append(current_q)
                context_lines = []
            current_q = {
                "number": int(m.group(1)),
                "question": m.group(2),
                "context": "",
                "options": []
            }
        elif current_q:
            # Could be context/instruction or a checkbox option
            if line.startswith("- ") or line.startswith("• ") or line.startswith("□ "):
                current_q["options"].append(line.lstrip("-•□ "))
            else:
                context_lines.append(line)

    # Save last question
    if current_q:
        current_q["context"] = "\n".join(context_lines).strip()
        questions.append(current_q)

    return questions


def generate_answers(
    form_text: str,
    org_name: str,
    org_context: str = "",
    language: str = "auto",
    verbose: bool = True,
) -> str:
    """
    Main entry point. Takes raw form text and generates answers for all questions.

    Args:
        form_text: Full text of the intake form (questions + instructions)
        org_name: Name of the target organisation (e.g. "M42", "Siemens")
        org_context: Extra context about the org (mission, focus areas, etc.)
        language: "en", "de", or "auto" (detect from form)
        verbose: Print progress

    Returns:
        Formatted string with all answers
    """
    client = anthropic.Anthropic()

    sector_pitch = detect_sector(org_name, form_text)

    # Detect language
    if language == "auto":
        german_signals = sum(1 for w in ["beschreib", "bitte", "dein", "startup", "lösung", "gründ"] if w in form_text.lower())
        lang = "German" if german_signals > 2 else "English"
    else:
        lang = "German" if language == "de" else "English"

    if verbose:
        print(f"\n  Target organisation: {org_name}")
        print(f"  Language: {lang}")
        print(f"  Sector context: {'Yes' if sector_pitch else 'Generic'}")

    system_prompt = f"""You are a strategic communications expert helping Aganthos, a cutting-edge AI startup, submit compelling answers to corporate partnership intake forms.

Your goal: make {org_name} genuinely excited to meet Aganthos. Each answer should:
- Be specific and concrete (reference real numbers, case studies, mechanisms)
- Connect Aganthos' capabilities directly to {org_name}'s strategic priorities
- Be confident but not salesy — let the technical substance speak
- Stay within typical form answer lengths (50–250 words per question, unless specified)
- Sound like a knowledgeable founder wrote it, not a marketing bot

Write all answers in {lang}.

{f"IMPORTANT ORGANISATIONAL CONTEXT FOR {org_name.upper()}:" + chr(10) + org_context if org_context else ""}

Here is Aganthos' full company profile:

{COMPANY_PROFILE}

{f"SECTOR-SPECIFIC CONTEXT (emphasise this):" + chr(10) + sector_pitch if sector_pitch else ""}

FORMATTING:
- Output each answer as: **Q[N]: [Question text]**\\nA: [Your answer]
- After all answers, add a **STRATEGIC NOTES** section with 2-3 brief tips on what to emphasise for this specific organisation
- If a question asks for a checkbox/multi-select, indicate which boxes Aganthos should tick and why
"""

    user_prompt = f"""Please answer all questions in the following intake form for {org_name}.
Use the Aganthos company profile to craft compelling, accurate answers tailored to {org_name}'s focus.

=== INTAKE FORM: {org_name.upper()} ===

{form_text}

=== END OF FORM ===

Generate a complete, submission-ready answer for every question/field. For open text fields, write full answer text. For multi-select/checkbox fields, clearly indicate which options to select."""

    if verbose:
        print(f"\n  Sending to Claude... ", end="", flush=True)

    result = _call_claude(system_prompt, user_prompt, max_tokens=4096)

    if verbose:
        print("done.")

    return result


def answer_single_question(
    question: str,
    form_context: str,
    org_name: str,
    org_context: str = "",
    language: str = "en",
    char_limit: int = 0,
) -> str:
    """
    Answer a single question interactively.
    Useful for one-off questions or refining specific answers.
    """
    sector_pitch = detect_sector(org_name, form_context)

    limit_instruction = f" Keep the answer under {char_limit} characters." if char_limit else ""

    system = f"""You are helping Aganthos answer a specific intake form question for {org_name}.
Be specific, compelling, and grounded in real data. Write in {"German" if language == "de" else "English"}.{limit_instruction}

{COMPANY_PROFILE}
{sector_pitch}"""

    user = f"""Form context: {form_context[:500] if form_context else 'N/A'}

Question: {question}

Write a compelling answer for Aganthos."""

    return _call_claude(system, user, max_tokens=1024)
