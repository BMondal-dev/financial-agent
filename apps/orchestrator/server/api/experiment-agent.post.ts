import { defineEventHandler, readBody } from "h3"
import { generateText, Output } from "ai"
import { google } from "@ai-sdk/google"
import { z } from "zod"
import { openai } from "@ai-sdk/openai"
import { chutes } from "../lib/openai"
import { anthropic } from "@ai-sdk/anthropic"

const FASTAPI = "http://localhost:8000"

function rankingMae(r: { mae_for_ranking?: number; mae?: number }) {
  return r.mae_for_ranking ?? r.mae ?? Number.POSITIVE_INFINITY
}

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
      // model: google("gemini-flash-latest"),
      model: openai("gpt-5.4-mini"),
      // model: chutes("moonshotai/Kimi-K2.6-TEE"),
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
  const model_type = body.model_type || "xgb"

  console.log(`\n[experiment-agent] 🚀 Starting prediction for target: ${target}, horizon: ${horizon}, model: ${model_type}`)

  // 1️⃣ Fetch Metadata
  // Using native fetch or $fetch (Nuxt/Nitro)
  const [metadata, candidates] = await Promise.all([
    $fetch(`${FASTAPI}/metadata/${target}`),
    $fetch(`${FASTAPI}/candidate-neighbors/${target}`)
  ])

  const sentimentData = await fetchRealSentiment(target)

  let graphNeighbors: any[] = []
  try {
    graphNeighbors = await $fetch(`${FASTAPI}/graph-neighbors/${target}`)
  } catch { graphNeighbors = [] }

  // Fetch past experiments for memory
  let pastExperiments: any[] = []
  try {
    pastExperiments = await $fetch(`${FASTAPI}/best-neighbors/${target}?horizon=${horizon}`)
  } catch { pastExperiments = [] }

  console.log(`[experiment-agent] 📊 Fetched metadata, candidates, sentiment (${sentimentData?.sentiment || 'None'}), graph neighbors.`)

  // 2️⃣ Ask LLM for experiments
  // We use generateObject here because you want a typed JSON response
  const { output } = await generateText({
    model: openai("gpt-5.4"),
    // model: anthropic("claude-haiku-4-5"),
    // model: google("gemini-3.1-pro-preview"),
    // model: chutes("moonshotai/Kimi-K2.6-TEE"),
    output: Output.object({
      schema: z.object({ // Property is 'schema', not 'output'
        experiments: z.array(
          z.object({
            explanation: z.string(),
            neighbors: z.array(z.string()).length(3)
          })
        )
      })
    }),
    prompt: `
      You are a quantitative finance researcher.
      Target stock: ${target}
      News Sentiment & Market Mood: ${JSON.stringify(sentimentData)}
      Metadata: ${JSON.stringify(metadata, null, 2)}
      Candidate neighbors: ${JSON.stringify(candidates)}
      Graph Neighbors: ${JSON.stringify(graphNeighbors)}
      Past Successful Tickers: ${JSON.stringify(pastExperiments)}

      Propose 3 different neighbor sets (3 stocks each) to forecast the target.
      Strategies can include: sector similarity, correlation similarity, 
      volatility regime similarity, consumption proxies, and cross-sector signals.
      Return diverse hypotheses. Incorporate sentiment/mood analysis and graph neighbors into your rationale.
    `,
  })

  const proposals = output.experiments
  const results: any[] = []

  console.log(`[experiment-agent] 🧠 AI generated ${proposals.length} proposals:`, JSON.stringify(proposals, null, 2));

  // 3️⃣ Run experiments
  console.log(`[experiment-agent] 🏃 Running experiments...`)
  for (const exp of proposals) {
    console.log(`[experiment-agent]    Testing neighbors: [${exp.neighbors.join(', ')}] ...`)
    const forecast: any = await $fetch(`${FASTAPI}/run-forecast`, {
      method: "POST",
      body: {
        target,
        neighbors: exp.neighbors,
        horizon,
        model_type,
        source: "experiment-agent",
        rationale: exp.explanation
      }
    })

    results.push({
      explanation: exp.explanation,
      neighbors: exp.neighbors,
      ...forecast // Fixed: 'forecast' is now typed/resolved as an object
    })
  }

  // 4️⃣ Select best experiment
  // Added a check to ensure results exist before sorting
  results.sort((a, b) => rankingMae(a) - rankingMae(b))

  const best = results[0]

  console.log(`[experiment-agent] ✅ Finished ${results.length} experiments. Best MAE: ${best?.mae}`)

  return {
    target,
    horizon,
    model_type,
    experiments: results,
    best
  }
})