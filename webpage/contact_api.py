#!/usr/bin/env python3
"""Aganthos contact form API. Receives POST, logs to file, sends email."""

import json
import os
import smtplib
import re
from datetime import datetime, timezone
from email.mime.text import MIMEText
from http.server import HTTPServer, BaseHTTPRequestHandler

LISTEN_PORT = 5000
MAILTO = "info@aganthos.com"
LOG_FILE = "/home/landingpage/contact_submissions.json"

def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

class ContactHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/contact":
            self.send_json(404, {"error": "not found"})
            return

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        if length > 10000:
            self.send_json(413, {"error": "too large"})
            return
        body = self.rfile.read(length)

        # Parse JSON
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "invalid json"})
            return

        # Extract and validate fields
        name = str(data.get("name", "")).strip()[:200]
        role = str(data.get("role", "")).strip()[:200]
        email = str(data.get("email", "")).strip()[:200]
        linkedin = str(data.get("linkedin", "")).strip()[:500]
        message = str(data.get("message", "")).strip()[:5000]

        if not name:
            self.send_json(400, {"error": "name is required"})
            return
        if not message:
            self.send_json(400, {"error": "message is required"})
            return
        if not email and not linkedin:
            self.send_json(400, {"error": "email or linkedin is required"})
            return
        if email and not is_valid_email(email):
            self.send_json(400, {"error": "invalid email format"})
            return

        # Log to file (permanent record)
        submission = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "role": role,
            "email": email,
            "linkedin": linkedin,
            "message": message,
        }
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(submission) + "\n")
        except Exception as e:
            print(f"[ERROR] Could not write log: {e}")

        # Send email
        try:
            subject = f"[aganthos.com] New inquiry from {name}"
            body_lines = [
                f"Name: {name}",
                f"Role: {role}" if role else None,
                f"Email: {email}" if email else None,
                f"LinkedIn: {linkedin}" if linkedin else None,
                "",
                "Message:",
                message,
                "",
                f"Received: {submission['timestamp']}",
            ]
            body_text = "\n".join(line for line in body_lines if line is not None)

            msg = MIMEText(body_text, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = f"Aganthos Website <noreply@aganthos.com>"
            msg["To"] = MAILTO
            if email:
                msg["Reply-To"] = email

            with smtplib.SMTP("localhost", 25) as smtp:
                smtp.send_message(msg)
        except Exception as e:
            print(f"[ERROR] Could not send email: {e}")
            # Submission is still logged, so not lost

        self.send_json(200, {"ok": True})

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "https://aganthos.com")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "https://aganthos.com")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        print(f"[{datetime.now().isoformat()}] {fmt % args}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    server = HTTPServer(("127.0.0.1", LISTEN_PORT), ContactHandler)
    print(f"Contact API listening on 127.0.0.1:{LISTEN_PORT}")
    server.serve_forever()
