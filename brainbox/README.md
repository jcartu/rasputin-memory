# BrainBox — Hebbian Procedural Memory

Learns agent behavior patterns using Hebbian association ("neurons that fire together, wire together").

## What it tracks
- **File co-occurrence** — files accessed together within 60s window
- **Command sequences** — commands run in sequence within 30s
- **Error→fix patterns** — which fixes resolve which errors

## Usage

```python
from brainbox import BrainBox
bb = BrainBox()

bb.record_access("config.yaml")
bb.record_access("deploy.sh")        # strengthens association

bb.suggest_files("config.yaml")       # → [("deploy.sh", 0.19)]
bb.suggest_next_command("git pull")   # → [("npm install", 0.1)]
bb.suggest_fix("ECONNREFUSED")        # → [("pm2 restart proxy", 0.19, 2)]
```

## CLI
```bash
python3 brainbox.py record-access "file.py"
python3 brainbox.py record-command "git status"
python3 brainbox.py record-error-fix "error msg" "fix command"
python3 brainbox.py suggest-files "file.py"
python3 brainbox.py suggest-commands "git pull"
python3 brainbox.py suggest-fix "error msg"
python3 brainbox.py stats
python3 brainbox.py test
```

## Storage
SQLite at `brainbox.db` (same dir). Override with `BRAINBOX_DB` env var.

## Hebbian Parameters
- Learning rate: 0.1 (asymptotic toward 1.0)
- Decay rate: 0.01 per decay cycle
- Co-occurrence window: 60s
- Sequence window: 30s
