# Agent Architecture Cleanup (Phase 3)

This note documents the orchestrator cleanup focused on **role separation** and **response consistency**.

## Goals

- Keep existing forecasting behavior and prompts mostly unchanged.
- Reduce endpoint duplication.
- Enforce stricter request/response handling.
- Prepare the codebase for future reliability and presentation upgrades.

## Current Role Mapping

The orchestrator now follows a role-oriented flow in `apps/orchestrator/server/lib/agent_pipeline.ts`:

- `curator(...)`
  - Loads metadata, candidate neighbors, graph neighbors, and past experiments.
- `plannerSingleRound(...)` / `plannerRound(...)`
  - Produces structured proposals from LLM outputs.
  - Applies proposal normalization and neighbor cleaning.
- `runner(...)`
  - Executes proposal batches via FastAPI `/run-forecast`.
  - Captures per-proposal failures without failing the whole endpoint.
- `reporterSingle(...)` / `reporterMultiRound(...)`
  - Returns stable response shapes.
  - Picks best result using ranking metric (`mae_for_ranking` fallback to `mae`).

## Endpoints Updated

- `apps/orchestrator/server/api/experiment-agent.post.ts`
  - Uses curator/planner/runner/reporter pipeline.
  - Validates request body via shared schema.
- `apps/orchestrator/server/api/experiment-multiple.post.ts`
  - Uses shared planner + runner per round.
  - Uses shared reporter for final payload.

## Shared Validation and Ranking

- Shared request schema:
  - `target` required.
  - `horizon` optional positive integer, capped at 30.
- Shared ranking:
  - `mae_for_ranking ?? mae ?? Infinity`.

## Why This Matters

- Lower maintenance cost: logic changes happen once in `agent_pipeline.ts`.
- Better reliability: malformed requests and proposal failures are handled consistently.
- Better presentation readiness: endpoint outputs are easier to explain and demo.

