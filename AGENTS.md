# AGENTS.md

Guidance for AI agents working in the Aion repository.

## What this project is

Aion is a **reusable memory library** for agents. It provides long-term memory outside the context window, focused on two memory types:

- **Episodic memory** — discrete events, indexed by recency and keywords.
- **Semantic memory** — distilled facts, indexed by meaning.

These types must stay **separate**: write both, retrieve separately. Do not merge episodic and semantic retrieval into a single undifferentiated store or query path.

Working memory (context window, agent state) and implicit memory (model weights, system prompts) are out of scope — they already exist elsewhere.

## What this file is (and is not)

This file provides **project context and guardrails**. It is **not** a build spec, task list, or permission to implement.

**Nothing in this repository should be built unless the owner assigns an explicit, scoped task.** If no task is given, discuss and answer questions only — do not write code, scaffold, or "helpfully" start implementation.

## Development approach

The owner designs. Agents implement.

1. The owner specifies what to build and how it should behave.
2. The agent implements that specification in small, scoped changes.
3. If anything required for the task is unspecified, **stop and ask** — do not guess, infer, or apply "reasonable defaults."
4. One concern per task. Minimal diff. No drive-by refactors or scope expansion.

### Agent may

- Implement behavior the owner has explicitly specified.
- Ask clarifying questions when requirements are missing or ambiguous.
- Read Mem0 docs or source **only when the assigned task says to use it as reference** — not to fill in unspecified design.

### Agent must not

- Make design decisions (architecture, algorithms, data models, APIs beyond what was specified).
- Invent memory approaches or do novel memory research in code.
- Start work that was not assigned.
- Choose storage backends, extraction strategies, deduplication logic, or security models without owner direction.
- Commit changes unless explicitly asked.

## Stated intent (not a build order)

The owner plans to work in a small, simple environment that separates concerns clearly:

| Component | Purpose |
|-----------|---------|
| **CLI** | Interactive loop; calls models via OpenRouter |
| **Memory** | Library with `add()`, `get()`, `update()`, `delete()` |
| **Storage** | JSON files initially; other backends later as needed |

First milestone target: **Mem0-like parity** — extract, store, and retrieve episodic and semantic memory separately through the CLI. This is the owner's goal, not an agent work queue.

**Strategy:** Reimplement Mem0-like behavior from scratch (Mem0 is a reference, not a dependency). Copy what works first; improve later.

## Open design areas (owner decides)

These are **not** agent decisions. Do not implement them until the owner specifies behavior:

- Memory extraction (what to extract, when, from what input)
- Storage format and layout beyond "JSON files to start"
- Deduplication strategy
- Security and access control
- Multi-user / multi-tenant support

## Tech stack

- **Language:** Python 3.12+
- **Dependencies:** stdlib + minimal third-party packages; `uv` or `pip`
- **Models:** OpenRouter (API key in `.env`, never committed or hardcoded)
- **Storage (v1):** Local JSON files

## Testing and commits

- Add tests **only when the owner asks**.
- Create git commits **only when the owner asks**.

## Non-goals

- Inventing a new memory paradigm before Mem0-like parity works.
- Building features, interfaces, or infrastructure the owner has not assigned (web UI, API server, multi-tenant, etc.).
- Using the Mem0 library as a dependency (reimplement instead).

## When stuck

Stop. State what is missing or ambiguous. Ask the owner. Do not proceed on assumptions.
