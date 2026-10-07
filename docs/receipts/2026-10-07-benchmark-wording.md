# Benchmark wording correction

Date: 2026-10-07. Owner: handoff_cleanup worker, reporting to portfolio root.

Objective: reconcile README throughput claims with the recorded dataset; no new benchmark or algorithm change.

Source: clean canonical C:/Users/PC/Documents/GitHub/turboquant, origin/main base 4de496d48bff9a7524016f60396aa374a999c887, sole writer codex/public-handoff-cleanup-20261007. Local dev workflow docs intentionally excluded. Allowed paths: README.md and this receipt. Root retains merge/release authority. No runtime release, paid GPU work or package publishing.

Evidence: both benchmarks/results_qwen2.5-3b-instruct.json and benchmark_results.json record Qwen2.5-3B at 3720 input tokens: FP16 2.4771576217480566 tok/s; TQ-4bit 7.447693637295166 tok/s, ratio 3.00654813884615. README table already matches the rounded values, but takeaway retained 3.5/6.1/74% and prose used 196% from rounded inputs. Short-context rows also demonstrate throughput reductions.

Change: remove universal 2x headline, use workload-scoped 3.01x from actual data consistently, cite dataset and explicitly describe shorter-context slowdowns. Remove misleading OOM/thrashing labels from successful measured headline runs. Stored data and algorithm unchanged.

Checks: parse both JSON datasets; assert identical relevant rows and derived ratio; verify corrected README values and absence of stale headline/takeaway/percentage. git diff --check. Documentation-only, no GPU/algorithm tests required.

Delivery: PR targets main. Root owns integration and GitHub source verification. Rollback: revert the scoped commit. No deployment or PyPI release required. Canonical checkout is borrowed for this sole-writer task; root owns restoration to dev after absorption. Existing untracked .playwright-mcp directory appeared when switching off dev (ignored there); preserved and excluded from commits.
