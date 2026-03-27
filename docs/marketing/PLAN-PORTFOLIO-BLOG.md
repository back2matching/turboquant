# Plan: Blog Section on matching.work

> Add a blog/research tab to the portfolio site. Markdown-based posts, brutalist aesthetic, first post is the TurboQuant benchmark article.

## Context

The portfolio at matching.work is a Next.js single-page app with a brutalist wireframe grid. Currently has About, Work, Contact sections. No blog. We want a place to publish technical research and benchmark findings that lives on our own domain — not just Reddit posts that disappear.

## What To Build

### Footer Navigation
Add a "blog" or "research" link to the footer (alongside email, twitter, telegram, github). Clicking it navigates to `/blog`.

### /blog Route
- Article listing page
- Shows post titles, dates, short descriptions
- Matches the existing brutalist wireframe aesthetic (dark bg, amber accents, monospace, no border-radius)
- Sorted by date, newest first

### /blog/[slug] Dynamic Routes
- Individual article pages
- Markdown content rendered with `next-mdx-remote` or `@next/mdx`
- Code syntax highlighting (for Python/bash code blocks)
- Clean reading experience — wide content area, monospace body text
- Back link to /blog listing

### Content System
- Posts stored as `.md` or `.mdx` files in `src/content/blog/` or similar
- Frontmatter: title, date, description, tags
- No CMS, no database — just markdown files in the repo
- Build-time rendering (static export compatible for Cloudflare Pages)

### First Post
The TurboQuant benchmark blog post from `docs/marketing/BLOG-POST-BENCHMARKS.md` — adapted for the site with proper frontmatter.

## Technical Approach

**Stack additions:**
- `next-mdx-remote` or `@next/mdx` for markdown rendering
- `rehype-highlight` or `shiki` for code syntax highlighting
- No other new dependencies needed

**File structure:**
```
src/
  app/
    blog/
      page.tsx          # Blog listing
      [slug]/
        page.tsx        # Individual post
  content/
    blog/
      turboquant-benchmarks.md   # First post
  lib/
    blog.ts             # getPostBySlug(), getAllPosts() utilities
```

**Styling:**
- Use existing CSS custom properties (--c-bg, --c-border, --c-accent, --c-text)
- Monospace body text (Geist Mono)
- Amber accent for links and headers
- Full-width content area (no grid columns on blog pages)
- BEM classes prefixed with `wf-blog-`

## Implementation Order

1. Add `/blog` route with static listing page
2. Add `/blog/[slug]` with MDX rendering
3. Style to match existing site aesthetic
4. Add footer nav link
5. Port benchmark blog post as first article
6. Test build + deploy to Cloudflare Pages

## Not In Scope (This Phase)
- RSS feed
- Search
- Tags/categories filtering
- Comments
- Newsletter signup
- Social sharing buttons
