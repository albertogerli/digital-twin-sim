# Deployment Guide

## Docker Compose (Local / VPS)

```bash
# 1. Clone and configure
git clone <repo-url>
cd digital-twin-sim
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY, DTS_API_KEYS, etc.

# 2. Build and start
docker-compose up -d --build

# 3. Verify
curl http://localhost/api/health
# {"status":"ok","postgres":true,"redis":true,...}
```

### SSL with Let's Encrypt

Add to `nginx.conf`:
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/yourdomain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain/privkey.pem;
    # ... rest of config
}
```

Or use a sidecar like `certbot` in docker-compose.

## VPS with systemd

### Backend Service

```ini
# /etc/systemd/system/dts-backend.service
[Unit]
Description=DigitalTwinSim Backend
After=network.target postgresql.service redis.service

[Service]
User=dts
WorkingDirectory=/opt/digital-twin-sim
EnvironmentFile=/opt/digital-twin-sim/.env
Environment=DTS_ENV=production
ExecStart=/opt/digital-twin-sim/.venv/bin/python run_api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Frontend Service

```ini
# /etc/systemd/system/dts-frontend.service
[Unit]
Description=DigitalTwinSim Frontend
After=network.target

[Service]
User=dts
WorkingDirectory=/opt/digital-twin-sim/frontend
Environment=NODE_ENV=production
Environment=PORT=3000
ExecStart=/usr/bin/node server.js
Restart=always

[Install]
WantedBy=multi-user.target
```

## Cloud Deployment

### AWS ECS / Fargate

- Backend: ECS service with Fargate, ALB routing `/api/*`
- Frontend: ECS service or S3 + CloudFront for static export
- PostgreSQL: RDS
- Redis: ElastiCache
- Secrets: AWS Secrets Manager → ECS task definition

### GCP Cloud Run

```bash
# Backend
gcloud run deploy dts-backend \
  --source . \
  --port 8000 \
  --set-env-vars "DTS_ENV=production" \
  --set-secrets "GOOGLE_API_KEY=google-api-key:latest"

# Frontend
cd frontend && npm run build
gcloud run deploy dts-frontend \
  --source . \
  --port 3000
```

- PostgreSQL: Cloud SQL
- Redis: Memorystore
- Use Cloud SQL Proxy for secure DB connections

## Environment Checklist

Before going to production:

- [ ] `DTS_ENV=production`
- [ ] `DTS_API_KEYS` or `DTS_KEY_MAP` configured
- [ ] `DTS_CORS_ORIGINS` restricted to actual domains
- [ ] `DTS_RATE_LIMIT_ENABLED=true`
- [ ] `DATABASE_URL` pointing to production PostgreSQL
- [ ] `REDIS_URL` pointing to production Redis
- [ ] `DTS_SENTRY_DSN_BACKEND` configured
- [ ] SSL/TLS termination configured
- [ ] `.env` not committed to git
- [ ] Firewall rules: only 80/443 public, 5432/6379 internal only
