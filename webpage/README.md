# Aganthos Website — Deployment & Infrastructure

## Server

- **Provider:** Hetzner
- **IP:** 78.47.141.140
- **OS:** Ubuntu 24.04.3 LTS
- **Hostname:** magentic

### Connect

```bash
ssh root@78.47.141.140
```

SSH key auth is configured. The key was added from Robert's MacBook via `ssh-copy-id`. SSH config entry (`~/.ssh/config`):

```
Host 78.47.141.140
  HostName 78.47.141.140
  User root
  ForwardAgent yes
```

---

## File Layout on Server

```
/home/landingpage/
  index.html                    # Main website (SPA, Tailwind CSS via CDN)
  logo.svg                      # Aganthos logo
  contact_api.py                # Contact form backend (Python, port 5000)
  contact_submissions.json      # Form submissions log (created on first submit)
```

---

## Nginx Configuration

Config file: `/etc/nginx/sites-enabled/landing-and-app`

### Domains

| Domain | What it serves | Backend |
|---|---|---|
| `aganthos.com` / `www.aganthos.com` | Landing page (static files) | `/home/landingpage/` |
| `aganthos.com/api/contact` | Contact form API | `proxy_pass → 127.0.0.1:5000` |
| `app.aganthos.com` | App (Docker) | `proxy_pass → 127.0.0.1:3000` |
| `tralala.aganthos.com` | GKE backend | `proxy_pass → 127.0.0.1:4000` |

### SSL

Managed by Certbot (Let's Encrypt). Certs at `/etc/letsencrypt/live/aganthos.com/`. Auto-renew via Certbot's systemd timer.

### Full Nginx Config

```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    return 404;
}

# 1) Root domain serves static files from /home/landingpage
server {
    server_name aganthos.com www.aganthos.com;

    root /home/landingpage;
    index index.html;

    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;

    location /api/contact {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Content-Type $content_type;
    }

    location / {
        try_files $uri $uri/ =404;
    }

    listen [::]:443 ssl ipv6only=on;
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/aganthos.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aganthos.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}

# 2) app.aganthos.com -> Docker app on 127.0.0.1:3000
server {
    server_name app.aganthos.com;
    client_max_body_size 50m;
    proxy_http_version 1.1;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
    proxy_connect_timeout 60s;
    proxy_buffering off;
    proxy_request_buffering off;

    location /socket.io/ {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
    }

    listen [::]:443 ssl;
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/aganthos.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aganthos.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}

# 3) tralala.aganthos.com -> 127.0.0.1:4000
server {
    server_name tralala.aganthos.com;
    client_max_body_size 50m;
    proxy_http_version 1.1;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
    proxy_connect_timeout 60s;
    proxy_buffering off;
    proxy_request_buffering off;

    location /socket.io/ {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
    }

    listen [::]:443 ssl;
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/aganthos.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aganthos.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}

# HTTP -> HTTPS redirects (managed by Certbot)
server {
    if ($host = www.aganthos.com) { return 301 https://$host$request_uri; }
    if ($host = aganthos.com) { return 301 https://$host$request_uri; }
    listen 80; listen [::]:80;
    server_name aganthos.com www.aganthos.com;
    return 404;
}
server {
    if ($host = app.aganthos.com) { return 301 https://$host$request_uri; }
    listen 80; listen [::]:80;
    server_name app.aganthos.com;
    return 404;
}
server {
    if ($host = tralala.aganthos.com) { return 301 https://$host$request_uri; }
    listen 80; listen [::]:80;
    server_name tralala.aganthos.com;
    return 404;
}
```

---

## Contact Form API

Self-hosted Python HTTP server. No third-party services — form data stays on the server.

### How it works

1. Browser POSTs JSON to `/api/contact`
2. Nginx proxies to `127.0.0.1:5000`
3. `contact_api.py` validates, logs to `contact_submissions.json`, sends email via local postfix
4. Returns JSON `{"ok": true}` or `{"error": "..."}`

### Systemd Service

File: `/etc/systemd/system/contact-api.service`

```ini
[Unit]
Description=Aganthos Contact Form API
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/landingpage/contact_api.py
Restart=always
RestartSec=5
WorkingDirectory=/home/landingpage

[Install]
WantedBy=multi-user.target
```

### Common Commands

```bash
# Check status
systemctl status contact-api

# Restart after code changes
systemctl restart contact-api

# View logs
journalctl -u contact-api -f

# View submissions
cat /home/landingpage/contact_submissions.json
```

### Email

Sent via local postfix (`localhost:25`). Postfix config: `myhostname = aganthos.com`, `inet_interfaces = loopback-only`. Emails go to `info@aganthos.com` (alias for `robert@aganthos.com` on Google Workspace).

---

## Deploying Updates

From local machine:

```bash
# Copy updated files to server
scp webpage/index.html webpage/logo.svg root@78.47.141.140:/home/landingpage/

# If contact_api.py changed, also restart the service:
scp webpage/contact_api.py root@78.47.141.140:/home/landingpage/
ssh root@78.47.141.140 "systemctl restart contact-api"

# If nginx config changed:
ssh root@78.47.141.140 "nginx -t && systemctl reload nginx"
```

---

## Quick Reference

| What | Command |
|---|---|
| SSH in | `ssh root@78.47.141.140` |
| Test nginx config | `nginx -t` |
| Reload nginx | `systemctl reload nginx` |
| Restart contact API | `systemctl restart contact-api` |
| View contact API logs | `journalctl -u contact-api -f` |
| View form submissions | `cat /home/landingpage/contact_submissions.json` |
| Renew SSL certs | `certbot renew` (auto via timer) |
| Check disk usage | `df -h /` |
