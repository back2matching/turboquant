# turboquant — Development Workflow

> How work gets done on this project. Adapted from FlockRun's workflow system.

---

## The Pipeline

Every task follows:

```
Read → Plan → Execute → Test → Docs → Commit
```

### 1. Read
- Read CLAUDE.md for current state
- Read relevant source files before touching anything
- Understand the blast radius

### 2. Execute
- Make the change
- For multi-file work, create a plan in `docs/plans/PLAN-<topic>.md`

### 3. Test
```bash
python -m pytest tests/ -v    # All 13 tests must pass
```

### 4. Docs (every commit)
| Doc | When to Update |
|-----|---------------|
| CLAUDE.md | If metrics changed (test count, version) |
| docs/ROADMAP.md | If features shipped or deferred |
| README.md | If user-facing features changed |

### 5. Commit
- `type: description` format (feat, fix, docs, refactor)
- Batch related work into one logical commit

---

## Git Strategy (Two Branches)

- **`main`** — public releases, PyPI publishes
- **`dev`** (if used) — daily work

### What's public vs private

| Public (goes to main/PyPI) | Private (dev-only or .claude/) |
|---------------------------|-------------------------------|
| Source code, README, LICENSE | CLAUDE.md (operating instructions) |
| Tests, examples, benchmarks | docs/ (internal research, plans, marketing) |
| pyproject.toml | .claude/ memory (credentials, server state) |

**NEVER commit to any branch:**
- API keys, tokens, passwords
- Server IPs or SSH keys
- Internal competitive analysis meant for eyes-only
- Marketing strategy docs with pricing/positioning details

These go in `.claude/` memory (not tracked by git) or stay on dev branch only.

---

## Related Projects

| Project | Repo | Role |
|---------|------|------|
| **FlockRun** | github.com/back2matching/flockrun | Parent project (agent runtime) |
| **turboquant-vectors** | github.com/back2matching/turboquant-vectors | Embedding privacy + compression (ACTIVE) |
| **kvcache-bench** | github.com/back2matching/kvcache-bench | KV cache benchmarking tool |
| **quant-sim** | github.com/back2matching/quant-sim | Model quantization simulator |

---

## First-Time Audit Prompt

Paste this when opening Claude Code in this repo for the first time:

```
Full end-to-end audit of this repo. Read every file — source code, tests, docs, config, README, CLAUDE.md, pyproject.toml. Don't skim, actually read the code.

Answer honestly:
1. What does this repo ACTUALLY do? (code, not README claims)
2. Current state? (version, test count, pass rate, last meaningful commit)
3. Finished, active, abandoned, or broken?
4. Docs: what exists, what's missing, what's wrong/stale?
5. Files that belong elsewhere, or missing files that belong here?
6. README accuracy — flag every claim not backed by code or tests
7. What breaks if someone pip installs this right now?
8. What would a new contributor need to know?

Then fix everything without asking:
- Update CLAUDE.md with real state
- Update README if claims don't match code
- Create/fix docs/ structure as needed
- Remove stale docs

Git rules:
- Work on main (or dev if it exists)
- docs/ is internal, may not reach public releases
- NEVER commit credentials, server IPs, API keys, internal strategy, or competitive analysis to ANY branch
- Those go in .claude/ memory only

Don't present options. Don't summarize. Fix it, commit, move on.
```
