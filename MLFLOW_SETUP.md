# MLflow Shared Tracking Setup

This document explains how to configure MLflow so that both your **GPU server**
and your **local laptop** write to the same experiment and can view each other's
runs in one shared UI.

---

## Why local `mlruns/` is not enough

By default MLflow writes to a local `./mlruns/` folder.  Each machine has its
own folder — runs logged on the GPU server are invisible on your laptop and
vice versa.  Copying the folder manually is error-prone and loses history.

---

## Why SQLite on a shared drive or S3 is unsafe

SQLite uses file-level locking.  When two processes (or two machines) write to
the same `.db` file at the same time — even if it is on a network share or
mounted S3 bucket — you get corrupted databases or silent data loss.
**Do not use a shared SQLite file as the MLflow backend store.**

---

## Recommended architecture

```
GPU server                  Local laptop
    |                           |
    |  MLFLOW_TRACKING_URI       |  MLFLOW_TRACKING_URI
    +---------> MLflow tracking server <---------+
                      |
          PostgreSQL (backend store)
                      |
              S3 bucket (artifacts)
```

All writes go through a single MLflow tracking server process.
Concurrent-write races are handled by the server; PostgreSQL handles
concurrent reads and writes safely.

---

## Step-by-step setup

### 1. Create a PostgreSQL database

Any PostgreSQL instance reachable by both machines works.
Options (roughly cheapest to most managed):

- **Self-hosted on the GPU server**: `sudo apt install postgresql`, then:
  ```sql
  CREATE DATABASE mlflow;
  CREATE USER mlflow_user WITH PASSWORD 'your_password';
  GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;
  ```
- **AWS RDS Free Tier** (db.t3.micro, 20 GB): reachable from anywhere via
  security-group rules; no maintenance required.
- **Supabase free tier**: PostgreSQL as a service, no credit card needed.

### 2. Create an S3 bucket for artifacts

```bash
aws s3 mb s3://your-mlflow-artifacts
```

Keep versioning off to avoid unnecessary storage costs.
Set a lifecycle rule to expire old artifacts if disk costs matter.

### 3. Start the MLflow tracking server (run this on the GPU server)

```bash
pip install mlflow psycopg2-binary boto3

mlflow server \
  --backend-store-uri postgresql://mlflow_user:your_password@localhost/mlflow \
  --default-artifact-root s3://your-mlflow-artifacts/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Keep it running in the background (tmux, screen, or a systemd unit).
Open port 5000 in the GPU server's firewall for your laptop's IP only.

### 4. Set environment variables on both machines

Create a `.env` file in the repo root (already in `.gitignore`):

```bash
# .env  — do NOT commit this file
MLFLOW_TRACKING_URI=http://<gpu-server-ip-or-hostname>:5000
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-west-1        # or wherever your bucket is
```

Load it before running training or evaluation:

```bash
# bash / zsh
export $(grep -v '^#' .env | xargs)
python -m src.train
python -m src.evaluate_forecast
```

Or set `tracking_uri` directly in `config.yaml` (less portable, keep
credentials out of version control):

```yaml
mlflow:
  experiment_name: cluster_first_stgnn_forecasting
  tracking_uri: "http://<gpu-server-ip>:5000"
```

### 5. Open the UI

From your laptop browser:

```
http://<gpu-server-ip>:5000
```

You will see all runs from both machines in one place.

---

## Minimal setup (no S3, local artifact storage on the GPU server)

If S3 is not available yet, use a local path for artifacts and only share
the tracking metadata via PostgreSQL:

```bash
mlflow server \
  --backend-store-uri postgresql://mlflow_user:your_password@localhost/mlflow \
  --default-artifact-root /home/your_user/mlflow_artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Artifacts are stored on the GPU server's disk.  Your laptop can view metric
plots in the UI but cannot download artifact files (predictions_test.npy,
figures, etc.) unless you copy them manually or mount the path over SSH/SSHFS.
Switch to S3 later without losing any run metadata.

---

## Docker Compose snippet (optional, for a self-contained setup)

```yaml
# docker-compose.yaml — run on the GPU server
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow_user
      POSTGRES_PASSWORD: your_password
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    depends_on:
      - postgres
    environment:
      AWS_ACCESS_KEY_ID: your_key_id
      AWS_SECRET_ACCESS_KEY: your_secret
      AWS_DEFAULT_REGION: eu-west-1
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow_user:your_password@postgres/mlflow
      --default-artifact-root s3://your-mlflow-artifacts/mlruns
      --host 0.0.0.0
      --port 5000
    ports:
      - "5000:5000"

volumes:
  pgdata:
```

Start with `docker compose up -d`.

---

## Config priority

The code resolves the tracking URI in this order (first non-empty wins):

1. `mlflow.tracking_uri` in `config.yaml`
2. `MLFLOW_TRACKING_URI` environment variable
3. MLflow default — local `./mlruns/` folder

This means you can override via the environment variable without editing
`config.yaml`, which is useful when the same config file is used on both
machines but the URI differs.

---

## Future orchestration repo

UPF assignment experiments belong in a separate MLflow experiment, suggested
name `forecast_driven_upf_assignment`.  That experiment lives in the future
orchestration/digital-twin repo and uses a separate set of runs with params
such as K, policy_name, alpha, thresholds, and power model version.
Do not log those runs here.
