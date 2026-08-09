---
name: pr-description
description: "Generates a concise, well-structured pull request description from the current branch's diff against main. Reads git history, summarises what changed and why, lists test steps, and pastes the result ready to copy into GitHub. Use for: writing PR descriptions, pull request summaries, PR body, PR writeup, PR template, describe my PR, what did I change."
---

# PR Description

Generate a pull request description for the current branch.

## Steps

1. **Gather context** — run these in parallel:
   ```
   git log main..HEAD --oneline
   git diff main...HEAD --stat
   git diff main...HEAD
   ```

2. **Analyse the diff** — identify:
   - *What* changed (files, components, endpoints, UI elements)
   - *Why* it changed (bug fix, feature, refactor, performance, chore)
   - Any behaviour changes visible to users or callers

3. **Write the description** using this template — output it as a markdown code block so the user can copy it directly:

   ```markdown
   ## Summary
   <!-- One sentence: what this PR does and why -->

   ### Changes
   - <!-- bullet per meaningful change; be specific, not "updated files" -->

   ### Test plan
   - [ ] <!-- concrete step to verify the change works -->
   - [ ] <!-- edge case or regression check if relevant -->

   ### Notes
   <!-- Optional: migration steps, known limitations, follow-up tickets, or nothing if clean -->
   ```

## Rules

- **No padding.** Omit any section that has nothing real to say (e.g. delete Notes if empty).
- **Why over what.** The diff already shows what changed; focus on the motivation.
- **Bullets, not prose.** Each change bullet starts with a verb: Add, Fix, Remove, Rename, Refactor, Update.
- **Test plan is runnable.** Each checklist item is a concrete action ("Open Screener tab → click ⬇ CSV → verify file downloads"), not a vague phrase ("test it").
- **No co-author lines** — those go in the commit, not the PR body.
- If the branch has no commits ahead of main, say so and stop.
