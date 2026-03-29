# Cron Jobs & Scheduled Maintenance

Recommended cron schedules for keeping the memory system healthy and up to date.

## Overview

| Job | Frequency | Script | Purpose |
|-----|-----------|--------|---------|
| Memory Autogen | Nightly (2am) | `tools/memory_autogen.py` | Regenerates MEMORY.md from top Qdrant results |
| Consolidator v4 | Weekly (Sun 3am) | `tools/memory_consolidator_v4.py` | Extracts facts from session transcripts |
| Fact Extractor | Every 4 hours | `tools/fact_extractor.py` | Structures facts from recent conversations |
| Memory Decay | Weekly (Sat 4am) | `tools/memory_decay.py` | Applies Ebbinghaus time decay to old memories |
| Dedup | Monthly (1st, 5am) | `tools/memory_dedup.py` | Removes near-duplicate vectors |
| Health Check | Every 30 min | `tools/memory_health_check.py` | Verifies all components are running |

## Example Crontab

```crontab
# Memory Autogen — regenerate hot context nightly
0 2 * * * cd /path/to/rasputin-memory && python3 tools/memory_autogen.py >> logs/autogen.log 2>&1

# Consolidator v4 — weekly transcript processing
0 3 * * 0 cd /path/to/rasputin-memory && python3 tools/memory_consolidator_v4.py >> logs/consolidator.log 2>&1

# Fact Extractor — every 4 hours
0 */4 * * * cd /path/to/rasputin-memory && python3 tools/fact_extractor.py >> logs/fact_extractor.log 2>&1

# Memory Decay — weekly score decay
0 4 * * 6 cd /path/to/rasputin-memory && python3 tools/memory_decay.py >> logs/decay.log 2>&1

# Dedup — monthly cleanup
0 5 1 * * cd /path/to/rasputin-memory && python3 tools/memory_dedup.py --execute >> logs/dedup.log 2>&1

# Health Check — every 30 minutes
*/30 * * * * cd /path/to/rasputin-memory && python3 tools/memory_health_check.py >> logs/health.log 2>&1
```

Replace `/path/to/rasputin-memory` with your actual repo path.

## Hot Context Directory

The `memory/hot-context/` directory is the output target for cron jobs that produce context for the AI agent:

- **memory_autogen.py** writes `MEMORY.md` and files in `memory/hot-context/`
- Files in `hot-context/` have a **24-hour TTL** — stale entries are ignored at session start
- Your AI agent should load `memory/hot-context/` files at the start of each session

### Directory Structure

```
memory/
├── hot-context/           # Auto-loaded at session start (24h TTL)
│   ├── recent-facts.md    # Latest extracted facts
│   ├── top-memories.md    # Highest-scored memories from Qdrant
│   └── ...
├── facts.jsonl            # Structured fact store
├── consolidation/         # Consolidator output and progress
└── ...
```

## Tips

- **First run:** Use `--all` flag for fact_extractor and consolidator to process historical sessions
- **Dry run dedup first:** Always run `python3 tools/memory_dedup.py --dry-run` before `--execute`
- **Log directory:** Create `logs/` in the repo root: `mkdir -p logs`
- **Monitor:** Check `logs/health.log` for service outages
