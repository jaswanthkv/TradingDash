---
name: improve-codebase-architecture
description: Scan a codebase for deepening opportunities, present them as a visual HTML report, then grill through whichever one you pick.
disable-model-invocation: true
---

# Improve Codebase Architecture

Surface architectural friction and propose **deepening opportunities** — refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability.

This command is self-contained: the vocabulary, principles, and process below are everything it needs. If the project has a `CONTEXT.md` (domain glossary) or `docs/adr/` (decision records), use them — `CONTEXT.md` gives names to good seams, and ADRs record decisions this command should not re-litigate. If either doesn't exist, skip it; don't block on their absence, and don't invent one just to satisfy this step.

## Vocabulary

Use these terms exactly in every suggestion — don't drift into "component," "service," "API," or "boundary."

- **Module** — a unit of code with an interface and an implementation.
- **Interface** — everything a caller must know to use a module: its signature plus the behavioral contract. This is the "cost" side of the module; callers pay for interface complexity every time they use it.
- **Depth** — the ratio of functionality hidden behind an interface to the complexity of that interface. A **deep** module does a lot through a small interface; a **shallow** module's interface is nearly as complex as what it does, so it isn't earning its keep as an abstraction.
- **Seam** — a place in the code where behavior can be substituted without editing the code at that point. Seams are where tests plug in fakes and where the system can be extended without modification.
- **Adapter** — a concrete implementation plugged into a seam (a fake, a different backend, a test double). **One adapter is a hypothetical seam; two adapters is a real one** — if only one implementation has ever used a seam, you don't actually know it generalizes.
- **Leverage** — how much downstream value a change at this point produces. High-leverage spots are where deepening pays off most; low-leverage spots aren't worth the churn.
- **Locality** — how close the code that decides something is to the code that acts on it. Pure functions extracted purely for testability, while the real bugs live in how/when they're called, have poor locality — the tests pass but they aren't testing the thing that breaks.

### Principles

- **The deletion test** — for anything you suspect is shallow, ask: if I deleted this and inlined its guts at the call site(s), would complexity concentrate somewhere useful, or just move around unchanged? "Concentrates" is the signal that the module is worth deepening rather than removing.
- **The interface is the test surface** — a module should be testable through its public interface. If the only way to test it is to reach into internals or mock out half its collaborators, the interface isn't actually where the tests live, and that's itself a shallowness signal.

## Process

### 1. Explore

**Scope before you scan — YAGNI.** Deepening a module pays off by making future changes to it easier, so put extra weight on the parts of the codebase that have recently changed. Decide *where* to look before you look:

- If the user named a direction — a module, a subsystem, a pain point — take it, and skip the inference below.
- Otherwise, walk back a good stretch of the commit history (`git log --oneline`) to find the codebase's hot spots — the files and areas that keep coming up — and let those paths pull your attention first. If the changes are scattered with no clear hot spot, widen the net.

Read `CONTEXT.md` and any ADRs in the area you're touching first, if they exist.

Then spawn a sub-agent to walk the codebase. Don't follow rigid heuristics — explore organically and note where you experience friction:

- Where does understanding one concept require bouncing between many small modules?
- Where are modules **shallow** — interface nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called (no **locality**)?
- Where do tightly-coupled modules leak across their seams?
- Which parts of the codebase are untested, or hard to test through their current interface?

Apply the **deletion test** to anything you suspect is shallow.

### 2. Present candidates as an HTML report

Write a self-contained HTML file to the OS temp directory so nothing lands in the repo. Resolve the temp dir from `$TMPDIR`, falling back to `/tmp` (or `%TEMP%` on Windows), and write to `<tmpdir>/architecture-review-<timestamp>.html` so each run gets a fresh file. Open it for the user — `xdg-open <path>` on Linux, `open <path>` on macOS, `start <path>` on Windows — and tell them the absolute path.

The report uses **Tailwind via CDN** for layout and styling, and **Mermaid via CDN** for diagrams where a graph/flow/sequence reliably communicates the structure. Mix Mermaid with hand-crafted CSS/SVG visuals — use Mermaid when relationships are graph-shaped (call graphs, dependencies, sequences), and hand-built divs/SVG when you want something more editorial (mass diagrams, cross-sections, collapse animations). Each candidate gets a **before/after visualisation**. Be visual.

For each candidate, render a card with:

- **Files** — which files/modules are involved
- **Problem** — why the current architecture is causing friction
- **Solution** — plain English description of what would change
- **Benefits** — explained in terms of locality and leverage, and how tests would improve
- **Before / After diagram** — side-by-side, custom-drawn, illustrating the shallowness and the deepening
- **Recommendation strength** — one of `Strong`, `Worth exploring`, `Speculative`, rendered as a badge

End the report with a **Top recommendation** section: which candidate you'd tackle first and why.

See [HTML-REPORT.md](HTML-REPORT.md) for the full HTML scaffold, diagram patterns, and styling guidance.

**Use `CONTEXT.md` vocabulary for the domain** (if it exists) — if it defines "Order," talk about "the Order intake module," not "the FooBarHandler" and not "the Order service." Use the vocabulary above for the architecture.

**ADR conflicts**: if a candidate contradicts an existing ADR, only surface it when the friction is real enough to warrant revisiting the ADR. Mark it clearly in the card (e.g. a warning callout: _"contradicts ADR-0007 — but worth reopening because…"_). Don't list every theoretical refactor an ADR forbids.

Do NOT propose interfaces yet. After the file is written, ask the user: "Which of these would you like to explore?"

### 3. Grilling loop

Once the user picks a candidate, walk the decision tree with them directly — don't just hand them a design, interrogate it together:

1. **Surface constraints** — ask what can't change (call sites that must keep working, data formats, external consumers, performance envelopes).
2. **Ask before assuming** — for anything ambiguous about the deepened module's responsibilities, ask a specific question rather than guessing; offer 2–3 concrete options when there's a real fork.
3. **Shape the interface** — propose the deepened module's interface (small, hides its complexity) and walk through 2–3 real call sites showing how they'd use it.
4. **Name the seam** — say explicitly what sits behind the new seam, and whether it currently has one adapter or two (hypothetical vs. real, per the vocabulary above).
5. **Check what survives** — identify which existing tests still pass unchanged, which need to move to the new interface, and which become unnecessary because the deepened module made the case they tested unreachable.
6. Only after the shape is agreed do you write the actual code change.

If the user wants to compare alternative interfaces for the deepened module before committing: spawn two sub-agents in parallel, each independently designing an interface for the same module with no visibility into the other's approach, then present both side by side with tradeoffs. Don't let one agent see the other's output before it's done — that's what keeps the comparison honest.

Side effects happen inline as decisions crystallize:

- **Naming a deepened module after a concept not in `CONTEXT.md`?** Add the term to `CONTEXT.md`. Create the file lazily if it doesn't exist — a short glossary entry (name, one-sentence definition, where it lives) is enough to start.
- **Sharpening a fuzzy term during the conversation?** Update `CONTEXT.md` right there.
- **User rejects the candidate with a load-bearing reason?** Offer an ADR, framed as: _"Want me to record this as an ADR so future architecture reviews don't re-suggest it?"_ Only offer when the reason would actually be needed by a future explorer to avoid re-suggesting the same thing — skip ephemeral reasons ("not worth it right now") and self-evident ones.
