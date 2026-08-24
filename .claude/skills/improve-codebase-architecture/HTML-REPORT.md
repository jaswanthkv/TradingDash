# HTML report scaffold

A self-contained HTML file — no build step, no local assets. Tailwind and Mermaid load from CDN; everything else is inline `<style>`/`<script>`.

## Base skeleton

```html
<!doctype html>
<html lang="en" class="scroll-smooth">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Architecture Review — <!-- project name --></title>
<script src="https://cdn.tailwindcss.com"></script>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "neutral", securityLevel: "loose" });
</script>
<style>
  body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .badge-strong        { @apply bg-emerald-100 text-emerald-800 border border-emerald-300; }
  .badge-worth-exploring { @apply bg-amber-100 text-amber-800 border border-amber-300; }
  .badge-speculative   { @apply bg-slate-100 text-slate-600 border border-slate-300; }
</style>
</head>
<body class="bg-slate-50 text-slate-900 leading-relaxed">

<header class="max-w-5xl mx-auto px-6 pt-12 pb-6">
  <h1 class="text-3xl font-bold tracking-tight">Architecture Review</h1>
  <p class="text-slate-500 mt-1"><!-- one-line scope: what was scanned and why --></p>
</header>

<main class="max-w-5xl mx-auto px-6 space-y-10 pb-16">
  <!-- one <section> per candidate, see "Candidate card" below -->
</main>

<footer class="max-w-5xl mx-auto px-6 pb-20">
  <!-- Top recommendation section, see below -->
</footer>

</body>
</html>
```

## Candidate card

Repeat this block per candidate. `id="candidate-N"` lets you deep-link to it from the top recommendation section.

```html
<section id="candidate-1" class="bg-white rounded-2xl border border-slate-200 shadow-sm p-8">
  <div class="flex items-start justify-between gap-4">
    <h2 class="text-xl font-semibold"><!-- candidate title, e.g. "Deepen the Order intake module" --></h2>
    <span class="badge-strong text-xs font-medium px-3 py-1 rounded-full whitespace-nowrap">Strong</span>
  </div>

  <p class="text-sm text-slate-500 mt-1 font-mono"><!-- Files: path/a.py, path/b.py --></p>

  <div class="grid md:grid-cols-2 gap-6 mt-6">
    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">Problem</h3>
      <p class="text-sm"><!-- why current architecture causes friction --></p>
    </div>
    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">Solution</h3>
      <p class="text-sm"><!-- plain-English description of the change --></p>
    </div>
  </div>

  <div class="mt-6">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">Benefits</h3>
    <ul class="text-sm list-disc list-inside space-y-1"><!-- locality / leverage / testability bullets --></ul>
  </div>

  <!-- optional: only when this candidate contradicts a recorded decision -->
  <div class="mt-6 bg-amber-50 border border-amber-200 text-amber-900 text-sm rounded-lg px-4 py-3">
    ⚠ Contradicts ADR-0007 — but worth reopening because <!-- reason -->.
  </div>

  <div class="mt-8">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-2">Before / After</h3>
    <div class="grid md:grid-cols-2 gap-6">
      <div class="border border-slate-200 rounded-xl p-4 bg-slate-50">
        <p class="text-xs font-medium text-slate-400 mb-2">BEFORE — shallow</p>
        <!-- Mermaid graph, or hand-built div/SVG mass diagram -->
        <pre class="mermaid">
graph TD
  Caller --> A[Small helper] --> B[Small helper] --> C[Small helper]
  Caller -.->|"has to know all three"| C
        </pre>
      </div>
      <div class="border border-emerald-200 rounded-xl p-4 bg-emerald-50">
        <p class="text-xs font-medium text-emerald-600 mb-2">AFTER — deep</p>
        <pre class="mermaid">
graph TD
  Caller --> D[Deepened module]
  D -.->|"hidden behind interface"| A2[Small helper]
  D -.-> B2[Small helper]
  D -.-> C2[Small helper]
        </pre>
      </div>
    </div>
  </div>
</section>
```

## Top recommendation

```html
<section class="bg-slate-900 text-white rounded-2xl p-8 mt-4">
  <h2 class="text-lg font-semibold mb-2">Top recommendation</h2>
  <p class="text-slate-200 text-sm">
    Start with <a href="#candidate-1" class="underline decoration-dotted">Candidate 1</a> —
    <!-- one or two sentences on why this one first: highest leverage, lowest risk, unblocks the others, etc. -->
  </p>
</section>
```

## Diagram guidance

- **Mermaid** for anything graph-shaped: call graphs, dependency chains, sequence diagrams of a request flowing through layers. Keep node labels short — Mermaid wraps badly inside Tailwind's constrained widths.
- **Hand-built SVG/CSS** for anything more editorial: a "mass" diagram showing interface size vs. implementation size as stacked bars, a cross-section of what's hidden behind a seam, or a before/after collapse animation (e.g. five boxes visually merging into one via a CSS transition on `:hover` or a scroll-triggered class toggle). Reach for this when the point is a *shape* or *proportion* (deep vs. shallow, small interface vs. large implementation) rather than a *relationship* — Mermaid is bad at proportion, good at relationship.
- Every diagram needs a one-line caption underneath explaining what to look at — don't make the user reverse-engineer the diagram's point.
- Keep the whole file under ~2000 lines. If there are more than ~6 candidates, group the weaker ones ("Speculative") into a collapsed `<details>` section instead of giving every candidate equal visual weight.
