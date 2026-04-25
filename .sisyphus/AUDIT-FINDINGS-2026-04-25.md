# Sensitive-Data Audit — sprint-2-integration @ 29bbcb7

**Date:** 2026-04-25
**Worktree:** `/home/josh/.openclaw/workspace/rasputin-memory-sprint2`
**Method:** Inline ripgrep / grep across tracked files (excluding `.git/`, `benchmarks/results/*.json`)
**Status:** Surfaced for user review. **No automatic redactions applied.**

The user requested findings only; classification (KEEP / SCRUB / REPLACE-WITH-PLACEHOLDER) is to be done by the user before Stage 2 cleanup commits.

---

## Section 1 — Definitely clean

| Category | Result |
|---|---|
| Personal email domains (gmail/protonmail/yandex/yahoo/outlook/hotmail/icloud) | None found |
| Bare "josh" prose mentions (excluding `/home/josh` paths) | None found |
| Hard-coded API keys with high-entropy patterns (`sk-…`, `csk-…`, `xai-…`, `gsk_…`, Bearer tokens) | None found |
| Hard-coded `API_KEY=` / `SECRET=` / `TOKEN=` assignments with real values | None found (only `.env.example` placeholders + `hooks/openclaw-mem/README.md` `your-deepseek-api-key` placeholder) |
| GPU UUIDs (`GPU-xxxxxxxx-…`) | None found |
| SSH / TLS private keys (`BEGIN RSA/OPENSSH/EC/DSA PRIVATE`) | None found |
| Telegram bot tokens / Slack tokens / Discord tokens | None found |
| `pyproject.toml` author/maintainer identity fields | Not set (no `authors = […]` block) |
| Git committer identity in tracked files | None found |
| Geographic operational keywords (Frankfurt, iGaming, VLESS, Reality, Hysteria, Sing-Box, Xray, casino) | None found |
| `.env.example` | All placeholders, no real values |
| Public IPs (non-RFC1918) in tracked files | None found |
| Production collection naming as a pure leak | `second_brain` used as default value across ~30 files. Not a secret per se but worth a decision (see Section 4) |

---

## Section 2 — Findings requiring user decision (KEEP / SCRUB / REPLACE)

### 2.1 GitHub username `jcartu` — **17 occurrences across 9 files**

Used as the canonical repo path. Files:

- `README.md:5` — CI badge
- `CONTRIBUTING.md:6` — clone URL
- `docs/SETUP.md:19` — clone URL
- `docs/GETTING_STARTED.md:21` — clone URL
- `CHANGELOG.md:270-277` — version compare links (×8)
- `SECURITY.md:15` — security advisory link
- `pyproject.toml:23-26` — Homepage / Documentation / Repository / Changelog URLs (×4)

**Note:** This IS the project's actual GitHub URL (where it will be archived). Removing it would break working links. The GH username `jcartu` is also publicly tied to the user's identity ("Joshua Cartu") via existing GitHub profile.

**Decision needed:**
- KEEP (project is already public on this GH user; archive will be at the same URL)
- SCRUB (replace with generic `OWNER/REPO` placeholder, breaks all links)
- REPLACE (move to a fresh GH org/user before archive)

---

### 2.2 Internal LAN hostnames + private IPs in benchmark scripts and experiment write-ups

| File | Line | Content |
|---|---|---|
| `benchmarks/run_ce_ab_test.sh` | 23, 27, 37 | `Arcstrider 192.168.1.41` + `CROSS_ENCODER_URL=http://192.168.1.41:9091/rerank` |
| `benchmarks/run_phase1.sh` | 17, 24 | `192.168.1.41:9091/rerank`, `Arcstrider 5090 GPU` |
| `benchmarks/run_phase2.sh` | 10 | `192.168.1.41:9091/rerank` |
| `docs/HYBRID-BRAIN.md` | 308 | `url = "http://192.168.1.41:9091/rerank"` |
| `experiments/2026-04-05_embedding_qwen3_vs_nomic.md` | 15 | `Sunbreaker (192.168.1.69, ~200ms/embed over LAN)` |
| `experiments/2026-04-05_qwen3_4096d.md` | 5, 15 | `Sunbreaker`, `Sunbreaker (192.168.1.69:11434)` |

`192.168.1.x` is RFC1918 — not a public-network leak. But:

- `Arcstrider` and `Sunbreaker` are personal machine names, mildly identifying.
- The IPs reveal the user runs the inference servers on a home LAN. Low operational risk; some users prefer to scrub.
- The experiment files are dated diary entries — they document the actual research methodology.

**Decision needed:**
- KEEP (RFC1918 IPs aren't sensitive; hostnames are colorful but not identifying)
- SCRUB to placeholders (`<RERANKER_HOST>:9091`, `<EMBED_HOST>:11434`) — preserves intent, drops trivia
- REPLACE Arcstrider/Sunbreaker → "Host A / Host B" or "GPU server / embedding server"

**Note on `experiments/`:** the user said "do not modify .sisyphus/ docs (they document the journey honestly)." Experiments aren't `.sisyphus/`, so they're in scope. But they have the same diary character. Worth confirming the user wants them touched.

---

### 2.3 `/home/josh/` paths in scripts and one doc

| File | Line | Content |
|---|---|---|
| `benchmarks/monitor_progress.py` | 6 | `CP = "/home/josh/.openclaw/workspace/rasputin-memory/benchmarks/results/locomo-leaderboard-checkpoint.json"` |
| `benchmarks/run_phase1.sh` | 4 | `cd /home/josh/.openclaw/workspace/rasputin-memory` |
| `benchmarks/run_phase2.sh` | 4 | `cd /home/josh/.openclaw/workspace/rasputin-memory` |
| `benchmarks/run_ce_ab_test.sh` | 4 | `cd /home/josh/.openclaw/workspace/rasputin-memory` |
| `docs/BACKLOG.md` | 59-60 | `python3 /home/josh/.openclaw/workspace/rasputin-memory/tools/qwen3_reranker_server.py 9091` and `Observed working directory: /home/josh/.openclaw/workspace/rasputin-memory` |

These are functional in scripts (the scripts assume a specific layout) but leak the username "josh" and the OpenClaw home path.

**Decision needed:**
- KEEP (scripts work as-is for the original developer; forks will need to adjust; harmless on a public repo)
- SCRUB to `$HOME/...` or `$REPO_ROOT` substitution (works for forks too)
- SCRUB only the doc occurrence in `BACKLOG.md` (lowest-effort hide of personal path)

---

### 2.4 `OpenClaw` brand references — **30+ occurrences across docs and hooks**

| File group | Notes |
|---|---|
| `docs/OPENCLAW-INTEGRATION.md` | Full integration doc (~285 lines) |
| `docs/AGENT-INTEGRATION.md:59-63` | Cross-references to OpenClaw |
| `docs/INTEGRATIONS.md:34-43` | OpenClaw section |
| `docs/GETTING_STARTED.md:114` | Mention |
| `docs/CONFIGURATION.md:116` | Port allocation note |
| `hooks/openclaw-mem/` | Whole hook directory + README |

OpenClaw is the user's personal agent framework / project name. References to it are functional documentation. It's a public-ish concept (referenced as `https://github.com/openclaw` in `hooks/openclaw-mem/README.md`).

**Decision needed:**
- KEEP (functional integration; OpenClaw is a thing the public-repo user cares about)
- SCRUB (rewrite as "your agent framework" — heavy edit, breaks specificity)
- KEEP but add a top-of-doc note that OpenClaw is a separate user project

---

### 2.5 GitHub usernames in `hooks/openclaw-mem/`

| File | Line | Content |
|---|---|---|
| `docs/OPENCLAW-INTEGRATION.md` | 38 | `git clone https://github.com/phenomenoner/openclaw-mem` |
| `hooks/openclaw-mem/README.md` | 22 | `git clone https://github.com/wenyupapa-sys/openclaw-mem.git ~/.openclaw/hooks/openclaw-mem` |

Two different GH usernames (`phenomenoner`, `wenyupapa-sys`) referenced for the same repo. Inconsistent — suggests stale / pseudonymous content.

**Decision needed:**
- KEEP both (preserves existing doc as-is)
- PICK ONE (canonicalize to whichever URL actually resolves)
- SCRUB (replace with placeholder owner)

---

### 2.6 `security@rasputin.to` email in `SECURITY.md`

`SECURITY.md:15` — vulnerability disclosure email.

The `rasputin.to` domain may or may not be registered/active. If it's a placeholder, real reporters will bounce. If it IS registered, it's a working contact channel.

**Decision needed:**
- KEEP (assume domain is set up)
- VERIFY domain status before archive
- REPLACE with the GH security advisory link only (drop email line)

---

### 2.7 Production collection name `second_brain`

Used as the default `QDRANT_COLLECTION` value across ~30 files (tools, tests, docs, config, scripts).

This is the user's actual production memory collection name. It's not a secret (Qdrant collection names aren't auth-relevant), and the test suite asserts on this exact string in 5+ tests. Changing it is a breaking change for anyone forking who tries to point at an existing instance.

**Decision needed:**
- KEEP (functional default; not a secret)
- RENAME to `rasputin_memory` (more neutral, matches project name; touches ~30 files including tests)
- KEEP in code, mention "default value is the original developer's collection name; override via `QDRANT_COLLECTION` env var" in README

---

### 2.8 Medical examples in fact-extraction prompts

| File | Line | Content (excerpt) |
|---|---|---|
| `tools/fact_extractor.py` | 217 | `- Health information: specific medications, dosages, conditions, doctors, dates` (instruction text in extractor prompt) |
| `tools/fact_extractor.py` | 337 | `[{{"fact": "The user increased their medication dosage from 5mg to 7.5mg on March 3"}}, …]` (few-shot example) |
| `tools/brain/amac.py` | 28 | `"The user's family member had a medical procedure at a major hospital." -> 10,9,10` (A-MAC quality-scoring example) |

These are example sentences in prompt templates. They're synthetic prompt-engineering examples, not real user data. Functional purpose is to teach the LLM what "high-importance personal facts" look like.

**Decision needed:**
- KEEP (synthetic prompt examples; demonstrate fact-extraction behavior)
- REPLACE with non-medical equivalents (work, travel, finance) — preserves intent without medical-domain specificity
- KEEP one, SCRUB others

---

### 2.9 Reference URL containing the word "affiliate" (per user's audit checklist)

Single hit:

- `docs/OPENCLAW-INTEGRATION.md:255` — `curl -s "http://localhost:7777/search?q=What+did+we+decide+about+the+product+affiliate+deal%3F&limit=5"`

Used as an example query string demonstrating the search API. The word "affiliate" here is generic ("affiliate deal" as a sample business question). No specific affiliate program named.

**Decision needed:**
- KEEP (generic business example, no operational link)
- REPLACE with neutral example (`What+did+we+decide+about+the+product+launch%3F`)

---

## Section 3 — Items the user explicitly listed but were not found

The user's audit checklist included these patterns. Confirming all clean:

- `joshua` (full name) — no hits
- `cartu` — only in `jcartu` GH username (Section 2.1)
- `iGaming`, `VLESS`, `Reality`, `Hysteria`, `Sing-Box`, `Xray` — no hits
- `Frankfurt` — no hits
- `honcho`, `cartu-proxy`, `wikiluck`, `casino-dashboard`, `enrich-daemon`, `rippers` — no hits except `Honcho` mentioned in `CHANGELOG.md:27-31` as a removed integration ("dead code — service decommissioned"). This is a *negative* mention in the changelog explicitly documenting Honcho was removed. Decision: probably KEEP (changelog entries are historically honest; documenting that something was removed is good practice).

---

## Section 4 — Other observations worth flagging

### 4.1 `.github/workflows/publish.yml`
Uses `${{ github.actor }}` for image publishing — this is the GH Actions runtime variable, not a hardcoded user. ✅ Clean.

### 4.2 `.github/dependabot.yml`
Standard config, no identity. ✅ Clean.

### 4.3 `pyproject.toml` package metadata
Notably has NO `authors = […]` block. Adding one for the public archive (e.g., `authors = [{name = "RASPUTIN Contributors"}]`) is optional. ✅ Currently neutral.

### 4.4 Hooks subproject (`hooks/openclaw-mem/`)
Self-contained; has its own README. The user may want to:
- KEEP it (functional integration, useful for OpenClaw users)
- DROP it (separates the public archive from the personal-framework integration)

The hook readme references separate GH repos (`phenomenoner/openclaw-mem`, `wenyupapa-sys/openclaw-mem`, `openclaw` org) — see Section 2.5.

---

## Section 5 — Recommended scope for Stage 2 cleanup commit (subject to user decision)

Based on findings, the **most defensible minimum scrub** for a public archive is:

1. **Replace `192.168.1.x` IPs in benchmark scripts and `docs/HYBRID-BRAIN.md`** with `${RERANKER_HOST}` / `${EMBED_HOST}` env-var placeholders (keeps scripts functional, removes home-LAN trivia).
2. **Replace `/home/josh/...` absolute paths in benchmark scripts and `docs/BACKLOG.md`** with `$REPO_ROOT` / relative paths.
3. **Verify `security@rasputin.to`** resolves OR drop the email line in `SECURITY.md`, leaving the GH security advisory link.
4. **Leave `jcartu`** — the archive lives at that GH path.
5. **Leave `OpenClaw` references** — functional integration docs.
6. **Leave `Arcstrider` / `Sunbreaker` hostnames** unless the user wants them genericized.
7. **Leave `second_brain` collection name** — functional default, not a secret.
8. **Leave medical prompt examples** — synthetic teaching examples, no real user data.

Maximum scrub (additional):
- Replace `Arcstrider` / `Sunbreaker` with neutral host names in experiments + scripts.
- Replace medical examples with neutral domain examples.
- Drop or rewrite `experiments/2026-04-05_*.md` (diary character).
- Add a generic `authors` block to `pyproject.toml`.

---

## Section 6 — User decisions required before Stage 2

For each numbered subsection in Section 2, the user should mark each finding:

- **K** = Keep as-is
- **S** = Scrub (delete entirely)
- **R** = Replace with placeholder
- **F** = Flag for further investigation

A simple decision matrix:

```
2.1  jcartu username                  [ K / S / R / F ]
2.2  Arcstrider/Sunbreaker + LAN IPs  [ K / S / R / F ]
2.3  /home/josh paths                 [ K / S / R / F ]
2.4  OpenClaw brand references        [ K / S / R / F ]
2.5  phenomenoner/wenyupapa-sys URLs  [ K / S / R / F ]
2.6  security@rasputin.to             [ K / S / R / F ]
2.7  second_brain collection name     [ K / S / R / F ]
2.8  Medical prompt examples          [ K / S / R / F ]
2.9  "affiliate" example query        [ K / S / R / F ]
```

Once the user provides decisions, Stage 2 will execute the scrub commit with the corresponding edits.
