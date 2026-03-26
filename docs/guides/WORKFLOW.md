# Development Workflow

> How work gets done. Adapted from FlockRun's ExecPlan system.

## The Pipeline

```
Read → Plan → Execute → Test → Docs → Commit
```

1. **Read** — CLAUDE.md for current state, docs/plans/ for active work
2. **Plan** — for multi-file work, create `docs/plans/PLAN-<topic>.md`
3. **Execute** — make the change
4. **Test** — `python -m pytest tests/ -v`
5. **Docs** — update CLAUDE.md and README if metrics/features changed
6. **Commit** — `type: description` format, batch related work

## ExecPlan System

For multi-hour autonomous work, create an ExecPlan in `docs/plans/`:

```markdown
# ExecPlan: <Title>
> One-line description. Created: <date>.

## Progress
- [x] (date) Milestone 1 done
- [ ] Milestone 2 in progress

## Milestones
### Milestone 1: <Name>
**What:** Deliverable. **Files:** Paths. **Verify:** Test command.
```

**Rules:** Never stop to ask. Skip MANUAL items. Update progress with timestamps. Commit at milestones. Verify before moving on.

Completed plans go to `docs/plans/archive/`.

## Git Strategy

Single branch (`main`). Push after meaningful milestones.

**NEVER commit credentials to any branch.** Use `.claude/` memory.

## First-Time Audit Prompt

```
Full end-to-end audit. Read every file. Answer: what does this do, current state,
finished/active/broken, docs missing/wrong, README accuracy. Then fix everything.
NEVER commit credentials or internal strategy. Don't ask, just fix.
```
