# Sub-plan 10 — CI workflow (lint / test / image build)

> Index: [`../plan.md`](../plan.md) · Depends on: [00 bootstrap](00-bootstrap.md) · Status: ☐

**Goal:** A single CI workflow at `.github/workflows/vrm-ci.yml` that runs on every PR or push under `projects/VRM-model/**`. On `main` (post-merge), it also builds and pushes the three Docker images to GHCR.

**Architecture:** Two jobs:
1. `test` — Ubuntu runner; uv install dev deps; ruff check + format; pyright; pytest with coverage.
2. `images` — only on push to `main`; uses `docker/build-push-action@v6` with QEMU + Buildx; pushes `ghcr.io/tech-sumit/vrm-{train,eval,dataprep}:latest` and `:sha-<short>`.

**Tech Stack:** GitHub Actions · `astral-sh/setup-uv@v3` · `docker/setup-buildx-action@v3` · `docker/login-action@v3` · `docker/build-push-action@v6`.

---

### Task 1: Repository-root setup (workflow directory + permission notes)

**Files:**
- Create: `.github/workflows/.gitkeep`

- [ ] **Step 1: Create the workflows directory at repo root**

```bash
mkdir -p /Users/sumitagrawal/CODE/sumit/n8n/.github/workflows
touch /Users/sumitagrawal/CODE/sumit/n8n/.github/workflows/.gitkeep
git add .github/workflows/.gitkeep
git commit -m "ci: create .github/workflows directory at repo root"
```

- [ ] **Step 2: Add a top-level `CODEOWNERS` for `projects/VRM-model/`** (so PRs auto-assign reviewers; optional)

```bash
cat >> /Users/sumitagrawal/CODE/sumit/n8n/.github/CODEOWNERS <<'EOF'
projects/VRM-model/ @tech-sumit
.github/workflows/vrm-*.yml @tech-sumit
EOF
git add .github/CODEOWNERS
git commit -m "ci: add CODEOWNERS for VRM-model paths"
```

---

### Task 2: `vrm-ci.yml` — lint/test/typecheck

**Files:**
- Create: `.github/workflows/vrm-ci.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/vrm-ci.yml`:

```yaml
name: vrm-ci

on:
  pull_request:
    paths:
      - "projects/VRM-model/**"
      - ".github/workflows/vrm-ci.yml"
  push:
    branches: [main]
    paths:
      - "projects/VRM-model/**"
      - ".github/workflows/vrm-ci.yml"
  workflow_dispatch:

concurrency:
  group: vrm-ci-${{ github.ref }}
  cancel-in-progress: true

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  test:
    name: lint + typecheck + test
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: projects/VRM-model/uv.lock

      - name: Set up Python 3.11
        run: uv python install 3.11

      - name: Install deps
        run: uv sync --extra dev

      - name: Lint (ruff check)
        run: uv run ruff check src tests

      - name: Format check (ruff format --check)
        run: uv run ruff format --check src tests

      - name: Type check (pyright)
        run: uv run pyright

      - name: Unit + integration tests (no GPU/network)
        run: uv run pytest -m "not gpu and not integration" --cov=src/vrm --cov-report=xml --cov-report=term-missing

      - name: Upload coverage
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ github.run_id }}
          path: projects/VRM-model/coverage.xml

  images:
    name: build & push images to GHCR
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: test
    runs-on: ubuntu-latest
    timeout-minutes: 90
    permissions:
      contents: read
      packages: write
      id-token: write
    strategy:
      fail-fast: false
      matrix:
        image:
          - { name: vrm-train, dockerfile: docker/train.Dockerfile }
          - { name: vrm-eval, dockerfile: docker/eval.Dockerfile }
          - { name: vrm-dataprep, dockerfile: docker/dataprep.Dockerfile }
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Compute short SHA
        id: sha
        run: echo "short=${GITHUB_SHA:0:7}" >> $GITHUB_OUTPUT

      - name: Build & push ${{ matrix.image.name }}
        uses: docker/build-push-action@v6
        with:
          context: projects/VRM-model
          file: projects/VRM-model/${{ matrix.image.dockerfile }}
          push: true
          tags: |
            ghcr.io/tech-sumit/${{ matrix.image.name }}:latest
            ghcr.io/tech-sumit/${{ matrix.image.name }}:sha-${{ steps.sha.outputs.short }}
          cache-from: type=gha,scope=${{ matrix.image.name }}
          cache-to: type=gha,scope=${{ matrix.image.name }},mode=max
          provenance: false   # avoids manifest list mismatch with RunPod's puller
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/vrm-ci.yml
git commit -m "ci: add vrm-ci workflow (uv lint/test/typecheck + GHCR image build on main)"
```

- [ ] **Step 3: Push and verify**

```bash
git push origin <feature-branch>
gh pr create --title "vrm: bootstrap project + CI" --body "Closes #XYZ"
gh pr checks --watch
```

Expected: `vrm-ci / test` PASSES (~3-5 min). `vrm-ci / images (matrix)` does NOT run on PRs (only on `main` push).

After merge to `main`, verify:
```bash
gh run list --workflow=vrm-ci.yml --limit 1
```

The `images` job should appear and complete (~10-20 min total for 3 images).

---

### Task 3: Confirm GHCR images are pullable

- [ ] **Step 1: From any host with Docker**

```bash
docker pull ghcr.io/tech-sumit/vrm-train:latest
docker pull ghcr.io/tech-sumit/vrm-eval:latest
docker pull ghcr.io/tech-sumit/vrm-dataprep:latest
```

If `unauthorized`, the package needs to be made public on GHCR (Settings → Packages → vrm-train → Change visibility), OR the puller needs a personal access token with `read:packages`. RunPod pulls the image; either make public or set `dockerCredentials` in the pod spec.

- [ ] **Step 2: Update `.env.example` if image visibility requires a token**

If kept private, add to `.env.example`:

```dotenv
GHCR_PULL_TOKEN=
```

And update `vrm.infra.runpod.PodSpec` to include the credentials in the pod payload (per RunPod docs: `containerRegistryAuth`).

- [ ] **Step 3: Commit any tweaks**

```bash
git add projects/VRM-model/.env.example projects/VRM-model/src/vrm/infra/runpod.py
git commit -m "vrm: support private GHCR pulls via containerRegistryAuth (if applicable)"
```

---

## Done when

- [ ] `vrm-ci / test` job goes green on a PR.
- [ ] `vrm-ci / images` job goes green on `main` and produces 3 images on GHCR.
- [ ] `docker pull ghcr.io/tech-sumit/vrm-train:latest` succeeds.
- [ ] Sub-plans 11 and 12 (CD workflows) can reference these images by `:latest` or `:sha-<short>`.
