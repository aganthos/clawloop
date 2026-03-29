#!/usr/bin/env python3
"""Demo script for clawloop + n8n integration.

Sends scripted customer support tickets through n8n, submits feedback,
waits for learning, then replays to show improvement.

Usage:
    python enterprise_clawloop/examples/n8n_cliproxy/demo.py [--n8n-url http://localhost:5678] [--clawloop-url http://localhost:8400]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def load_tickets(path: str = "enterprise_clawloop/config/demo_tickets.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def send_ticket(client: httpx.Client, n8n_webhook_url: str, ticket: dict) -> dict:
    resp = client.post(n8n_webhook_url, json={"message": ticket["message"]}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def wait_for_idle(client: httpx.Client, clawloop_url: str, timeout: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"{clawloop_url}/state")
        state = resp.json()
        if state["learning_status"] == "idle" and state["playbook_version"] > 0:
            return
        time.sleep(1.0)
    raise TimeoutError("Learning did not complete within timeout")


def main():
    parser = argparse.ArgumentParser(description="clawloop + n8n demo")
    parser.add_argument("--n8n-url", default="http://localhost:5678")
    parser.add_argument("--clawloop-url", default="http://localhost:8400")
    parser.add_argument("--webhook-path", default="/webhook/support")
    parser.add_argument("--tickets", default="enterprise_clawloop/config/demo_tickets.json")
    args = parser.parse_args()

    webhook_url = f"{args.n8n_url}{args.webhook_path}"
    tickets = load_tickets(args.tickets)
    client = httpx.Client()

    print("=" * 60)
    print("ROUND 1 — Baseline (no learning)")
    print("=" * 60)

    for ticket in tickets:
        print(f"\n[{ticket['id']}] {ticket['message'][:60]}...")
        try:
            result = send_ticket(client, webhook_url, ticket)
            response_text = result.get("response", result.get("output", str(result)))
            print(f"  -> {str(response_text)[:100]}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    state = client.get(f"{args.clawloop_url}/state").json()
    print(f"\nEpisodes collected: {state['metrics']['episodes_collected']}")

    print("\nSubmitting feedback (thumbs down on T2-T5)...")
    # Note: feedback needs episode IDs from the server
    # In real demo, get them from dashboard or state endpoint

    print("\nWaiting for learning to complete...")
    try:
        wait_for_idle(client, args.clawloop_url, timeout=120.0)
    except TimeoutError:
        print("ERROR: Learning timed out")
        sys.exit(1)

    state_after = client.get(f"{args.clawloop_url}/state").json()
    print(f"\nPlaybook version: {state_after['playbook_version']}")
    print(f"Playbook entries: {len(state_after['playbook_entries'])}")
    for entry in state_after["playbook_entries"]:
        print(f"  * {entry['content'][:80]}")

    print("\n" + "=" * 60)
    print("ROUND 2 — After Learning")
    print("=" * 60)

    for ticket in tickets[1:]:
        print(f"\n[{ticket['id']}] {ticket['message'][:60]}...")
        try:
            result = send_ticket(client, webhook_url, ticket)
            response_text = result.get("response", result.get("output", str(result)))
            print(f"  -> {str(response_text)[:100]}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    final = client.get(f"{args.clawloop_url}/state").json()
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Playbook version: {final['playbook_version']}")
    print(f"Entries learned: {len(final['playbook_entries'])}")
    print(f"\nBefore prompt:\n  {state['system_prompt'][:200]}")
    print(f"\nAfter prompt:\n  {final['system_prompt'][:200]}")

    client.close()


if __name__ == "__main__":
    main()
