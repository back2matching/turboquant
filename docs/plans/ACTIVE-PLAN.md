# ACTIVE PLAN: TurboQuant v0.2.0 Launch & Visibility

> Created: 2026-03-26
> Status: IN PROGRESS

## Progress

- [x] (2026-03-26 12:00) Benchmark campaign complete — 45 data points, 4 models, RTX 4080
- [x] (2026-03-26 12:30) Compressed index storage shipped (v0.2.0)
- [x] (2026-03-26 13:05) Documentation accuracy sweep — all refs updated for v0.2.0
- [x] (2026-03-26 13:15) 4-bit nibble packing — halves index storage
- [x] (2026-03-26 13:20) Marketing folder created — strategy, blog post, Reddit drafts, Twitter thread
- [x] (2026-03-26 13:40) v0.2.0 published to PyPI — https://pypi.org/project/turboquant/0.2.0/
- [x] (2026-03-26 13:41) PyPI API token "turboquant-upload" needs rotation (exposed in session log)
- [ ] Portfolio blog feature on matching.work
- [ ] Publish Reddit/social posts
- [ ] Re-benchmark StableLM with v0.2.0 compressed storage

## Phase 1: Code & Publish (COMPLETE)

All code work is done and published:
- v0.2.0 on PyPI with compressed index storage + nibble packing
- 17 tests passing
- 8 commits on main, all pushed

## Phase 2: Visibility (NEXT)

### Step 1: Portfolio blog on matching.work
- Detailed plan in `docs/marketing/PLAN-PORTFOLIO-BLOG.md`
- Add `/blog` route to portfolio site (Next.js)
- Port benchmark blog post as first article
- **Repo:** `C:\Users\PC\Documents\GitHub\portfolio`
- **Deploy:** Cloudflare Pages (auto on push to main)

### Step 2: Post content
- Reddit posts drafted in `docs/marketing/REDDIT-POSTS.md`
- Twitter thread drafted in same file
- Post to r/LocalLLaMA first (core audience), then r/MachineLearning, r/programming
- Link to blog post on matching.work (once live)

### Step 3: Re-benchmark StableLM
- StableLM data is from v0.1.0 (dequantized storage)
- v0.2.0 compressed storage may show different results
- Run: `python benchmarks/benchmark_kv.py --model stabilityai/stablelm-2-1_6b-chat --context "512,1024,2048,4096"`

## Phase 3: Future Optimizations (PARKED)

These are possible next steps, not planned:
- 3-bit sub-byte packing (8 values into 3 bytes)
- Wire CUDA kernel into main classes
- Streaming server (FastAPI)
- vLLM PagedAttention support
- Perplexity benchmarks (WikiText-2)

## Security Note

The PyPI API token "turboquant-upload" was exposed in a Claude Code session log on 2026-03-26. It should be rotated via https://pypi.org/manage/account/ — delete the old token and create a new one before next upload.

## Key Files

| File | What |
|------|------|
| `docs/marketing/STRATEGY.md` | Positioning, audience, proof points |
| `docs/marketing/BLOG-POST-BENCHMARKS.md` | Full benchmark blog article |
| `docs/marketing/REDDIT-POSTS.md` | Reddit + Twitter drafts |
| `docs/marketing/PLAN-PORTFOLIO-BLOG.md` | Portfolio blog feature spec |
| `benchmarks/benchmark_results.json` | All 45 data points |
