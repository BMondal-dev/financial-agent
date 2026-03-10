import { defineEventHandler, readBody } from "h3"
import { generateText, Output } from "ai"
import { z } from "zod"
import { google } from "@ai-sdk/google"

const FASTAPI = "http://localhost:8000"

async function runForecast(target: string, neighbors: string[], horizon: number) {
  return await $fetch(`${FASTAPI}/run-forecast`, {
    method: "POST",
    body: { target, neighbors, horizon }
  })
}

const ExperimentSchema = z.object({
  experiment_id: z.string(),
  rationale: z.string(),
  neighbors: z.array(z.string()).max(3)
})

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const target = body.target
  const horizon = body.horizon || 5

  // --- Data Loading ---
  const [metadata, candidates] = await Promise.all([
    $fetch(`${FASTAPI}/metadata/${target}`),
    $fetch(`${FASTAPI}/candidate-neighbors/${target}`)
  ])

  let graphNeighbors: any[] = []
  try {
    graphNeighbors = await $fetch(`${FASTAPI}/graph-neighbors/${target}`)
  } catch { graphNeighbors = [] }

  let pastExperiments: any[] = []
  try {
    pastExperiments = await $fetch(`${FASTAPI}/best-neighbors/${target}`)
  } catch { pastExperiments = [] }

  // -----------------------------------
  // ROUND 1 — Exploration
  // -----------------------------------
  const { output: round1 } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.array({ element: ExperimentSchema }),
    system: "You are a quant researcher. Return ONLY a JSON array of objects.",
    prompt: `
      Target: ${target}
      Metadata: ${JSON.stringify(metadata)}
      Candidates: ${JSON.stringify(candidates)}
      Graph Neighbors: ${JSON.stringify(graphNeighbors)}
      Past Successful Tickers: ${JSON.stringify(pastExperiments)}
      
      Propose 3 DIFFERENT neighbor sets. Each must have:
      - experiment_id: unique string (e.g. "r1_exp1")
      - rationale: brief explanation
      - neighbors: array of up to 3 ticker strings
      
      Reference past successful experiments where relevant.
    `
  })

  const round1Results: any[] = []
  for (const exp of round1) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon)
      round1Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round1Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  const bestRound1 = [...round1Results].sort((a, b) => a.mae - b.mae)[0]

  // -----------------------------------
  // ROUND 2 — Refinement
  // -----------------------------------
  const { output: round2 } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.array({ element: ExperimentSchema }),
    system: "You are a quant researcher. Return ONLY a JSON array of objects.",
    prompt: `
      Best from Round 1: ${JSON.stringify(bestRound1)}
      All Round 1 Results: ${JSON.stringify(round1Results)}
      Past Successful Tickers: ${JSON.stringify(pastExperiments)}
      Graph Neighbors: ${JSON.stringify(graphNeighbors)}
      
      Refine 3 NEW neighbor sets building on what worked. Each must have:
      - experiment_id: unique string (e.g. "r2_exp1")
      - rationale: brief explanation
      - neighbors: array of up to 3 ticker strings
      
      Do NOT replicate failed patterns from Round 1.
    `
  })

  const round2Results: any[] = []
  for (const exp of round2) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon)
      round2Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round2Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  const bestRound2 = [...round2Results].sort((a, b) => a.mae - b.mae)[0]

  // -----------------------------------
  // ROUND 3 — Mutation Search
  // -----------------------------------
  const { output: round3 } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.array({ element: ExperimentSchema }),
    system: "You are a quant researcher. Return ONLY a JSON array of objects.",
    prompt: `
      Best from Round 1: ${JSON.stringify(bestRound1)}
      Best from Round 2: ${JSON.stringify(bestRound2)}
      Historical Context: ${JSON.stringify(pastExperiments)}
      
      Generate 3 mutated combinations using cross-sector signals. Each must have:
      - experiment_id: unique string (e.g. "r3_exp1")
      - rationale: brief explanation
      - neighbors: array of up to 3 ticker strings
      
      Be creative — try unconventional cross-sector pairings that could surface hidden correlations.
    `
  })

  const round3Results: any[] = []
  for (const exp of round3) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon)
      round3Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round3Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  // --- Final Best ---
  const finalResults = [...round1Results, ...round2Results, ...round3Results]
    .filter(r => r.mae !== Infinity)
    .sort((a, b) => a.mae - b.mae)

  return {
    target,
    horizon,
    round1: round1Results,
    round2: round2Results,
    round3: round3Results,
    best: finalResults[0] ?? null
  }
})