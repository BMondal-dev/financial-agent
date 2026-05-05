import { defineEventHandler, readBody } from "h3"
import { generateText, Output } from "ai"
import { z } from "zod"
import { google } from "@ai-sdk/google"
import { chutes } from "../lib/openai"

const FASTAPI = "http://localhost:8000"

function rankingMae(r: { mae_for_ranking?: number; mae?: number }) {
  return r.mae_for_ranking ?? r.mae ?? Number.POSITIVE_INFINITY
}

async function runForecast(
  target: string,
  neighbors: string[],
  horizon: number,
  meta: { experiment_id: string; rationale: string; orchestrator_round: number }
) {
  return await $fetch(`${FASTAPI}/run-forecast`, {
    method: "POST",
    body: {
      target,
      neighbors,
      horizon,
      source: "experiment-multiple",
      experiment_id: meta.experiment_id,
      rationale: meta.rationale,
      orchestrator_round: meta.orchestrator_round
    }
  })
}

const ExperimentSchema = z.object({
  experiment_id: z.string(),
  rationale: z.string(),
  neighbors: z.array(z.string()).max(3)
})

const SentimentSchema = z.object({
  sentiment: z.enum(["Bullish", "Bearish", "Neutral"]),
  semantic_sentiment: z.string(),
  market_mood_index: z.number().describe("Float between 0.00 and 1.00. 1.00 is extremely Bullish (optimistic), 0.00 is extremely Bearish (pessimistic), and 0.50 is Neutral.")
})

async function fetchRealSentiment(target: string) {
  try {
    const rssResponse = await $fetch<string>(`https://news.google.com/rss/search?q=${target}+stock&hl=en-US&gl=US&ceid=US:en`);
    // Simple regex to grab the first 10 <title> contents, skipping the main feed title
    const titles = [...rssResponse.matchAll(/<title>(.*?)<\/title>/g)].map(m => m[1]).filter(t => !t.includes('Google News')).slice(0, 10);

    if (titles.length === 0) return null;

    const { output } = await generateText({
      model: google("gemini-flash-latest"),
      output: Output.object({
        schema: SentimentSchema
      }),
      system: "You are a quant researcher analyzing the sentiment and market mood for a given stock based on recent news headlines. Provide a sentiment label, a concise semantic description of the news, and a market mood index (0.00 to 1.00).",
      prompt: `Analyze the following news headlines for ${target}:\n${titles.join("\n")}`
    });

    return output;
  } catch (e) {
    console.error("Failed to fetch or generate sentiment:", e);
    return null;
  }
}

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const target = body.target
  const horizon = body.horizon || 5

  // --- Data Loading ---
  const [metadata, candidates] = await Promise.all([
    $fetch(`${FASTAPI}/metadata/${target}`),
    $fetch(`${FASTAPI}/candidate-neighbors/${target}`)
  ])

  const sentimentData = await fetchRealSentiment(target)

  let graphNeighbors: any[] = []
  try {
    graphNeighbors = await $fetch(`${FASTAPI}/graph-neighbors/${target}`)
  } catch { graphNeighbors = [] }

  let pastExperiments: any[] = []
  try {
    pastExperiments = await $fetch(`${FASTAPI}/best-neighbors/${target}?horizon=${horizon}`)
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
      News Sentiment & Market Mood: ${JSON.stringify(sentimentData)}
      Metadata: ${JSON.stringify(metadata)}
      Candidates: ${JSON.stringify(candidates)}
      Graph Neighbors: ${JSON.stringify(graphNeighbors)}
      Past Successful Tickers: ${JSON.stringify(pastExperiments)}
      
      Propose 3 DIFFERENT neighbor sets. Each must have:
      - experiment_id: unique string (e.g. "r1_exp1")
      - rationale: brief explanation (incorporating sentiment/mood analysis if applicable)
      - neighbors: array of up to 3 ticker strings
      
      Reference past successful experiments where relevant. Consider sentiment alignment or divergence when selecting neighbors.
    `
  })

  const round1Results: any[] = []
  for (const exp of round1) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon, {
        experiment_id: exp.experiment_id,
        rationale: exp.rationale,
        orchestrator_round: 1
      })
      round1Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round1Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  const bestRound1 = [...round1Results].sort((a, b) => rankingMae(a) - rankingMae(b))[0]

  // -----------------------------------
  // ROUND 2 — Refinement
  // -----------------------------------
  const { output: round2 } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.array({ element: ExperimentSchema }),
    system: "You are a quant researcher. Return ONLY a JSON array of objects.",
    prompt: `
      Target: ${target}
      News Sentiment & Market Mood: ${JSON.stringify(sentimentData)}
      Best from Round 1: ${JSON.stringify(bestRound1)}
      All Round 1 Results: ${JSON.stringify(round1Results)}
      Past Successful Tickers: ${JSON.stringify(pastExperiments)}
      Graph Neighbors: ${JSON.stringify(graphNeighbors)}
      
      Refine 3 NEW neighbor sets building on what worked. Each must have:
      - experiment_id: unique string (e.g. "r2_exp1")
      - rationale: brief explanation (incorporating sentiment/mood analysis)
      - neighbors: array of up to 3 ticker strings
      
      Do NOT replicate failed patterns from Round 1. Consider market mood alignment.
      
      Regime Risk Check: Pay attention to 'cv_worst_split_mae' and its date range. 
      If a neighbor set has a low average MAE but a high worst-split MAE during a specific period, 
      it may be fragile. Prefer sets with consistent performance across time.
    `
  })

  const round2Results: any[] = []
  for (const exp of round2) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon, {
        experiment_id: exp.experiment_id,
        rationale: exp.rationale,
        orchestrator_round: 2
      })
      round2Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round2Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  const bestRound2 = [...round2Results].sort((a, b) => rankingMae(a) - rankingMae(b))[0]

  // -----------------------------------
  // ROUND 3 — Mutation Search
  // -----------------------------------
  const { output: round3 } = await generateText({
    model: chutes("zai-org/GLM-5-Turbo"),
    output: Output.array({ element: ExperimentSchema }),
    system: "You are a quant researcher. Return ONLY a JSON array of objects.",
    prompt: `
      Target: ${target}
      News Sentiment & Market Mood: ${JSON.stringify(sentimentData)}
      Best from Round 1: ${JSON.stringify(bestRound1)}
      Best from Round 2: ${JSON.stringify(bestRound2)}
      Historical Context: ${JSON.stringify(pastExperiments)}
      
      Generate 3 mutated combinations using cross-sector signals. Each must have:
      - experiment_id: unique string (e.g. "r3_exp1")
      - rationale: brief explanation (incorporating sentiment/mood analysis)
      - neighbors: array of up to 3 ticker strings
      
      Be creative — try unconventional cross-sector pairings that could surface hidden correlations. Use sentiment or market mood context to find counter-intuitive matches.
      
      Regime Risk Check: Avoid neighbor sets that historically failed during specific market regimes 
      (check 'cv_worst_split_test_date_start/ end'). Prioritize cross-sector pairs that show stable CV metrics.
    `
  })

  const round3Results: any[] = []
  for (const exp of round3) {
    try {
      const result: any = await runForecast(target, exp.neighbors, horizon, {
        experiment_id: exp.experiment_id,
        rationale: exp.rationale,
        orchestrator_round: 3
      })
      round3Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, ...result })
    } catch (e) {
      round3Results.push({ experiment_id: exp.experiment_id, rationale: exp.rationale, neighbors: exp.neighbors, mae: Infinity, error: String(e) })
    }
  }

  // --- Final Best ---
  const finalResults = [...round1Results, ...round2Results, ...round3Results]
    .filter(r => r.mae !== Infinity && typeof r.mae === "number")
    .sort((a, b) => rankingMae(a) - rankingMae(b))

  const best = finalResults[0] ?? null

  return {
    target,
    horizon,
    round1: round1Results,
    round2: round2Results,
    round3: round3Results,
    best
  }
})
