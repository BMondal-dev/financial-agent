import { readFile } from "fs/promises"
import { join } from "path"
import { generateText } from "ai"
import { google } from "@ai-sdk/google"
import { defineEventHandler, readBody } from "h3"

// Define the type for the forecast response
interface ForecastResponse {
  mae: number;
  predicted_return: number;
}

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { target, horizon = 5 } = body

  if (!target) {
    return { error: "target is required" }
  }

  // ---- Load metadata ----
  const metadataPath = join(process.cwd(), "../../data/metadata.json")
  const metadataRaw = await readFile(metadataPath, "utf-8")
  const metadata = JSON.parse(metadataRaw)

  const targetMeta = metadata[target]

  if (!targetMeta) {
    return { error: `Target ${target} not found` }
  }

  // ---- Build Prompt ----
  const prompt = `
You are a financial data selection agent.

Target stock: ${target}
Sector: ${targetMeta.sector}
Volatility (30d): ${targetMeta.volatility_30d}

Top Correlated Stocks:
${JSON.stringify(targetMeta.top_correlated, null, 2)}

Same Sector Stocks:
${JSON.stringify(targetMeta.same_sector, null, 2)}

Task:
Generate 3 different dataset enrichment proposals.
Each proposal must include:
- explanation (short reasoning)
- neighbors (array of up to 3 stock symbols)

Output ONLY valid JSON in this format:

{
  "proposals": [
    {
      "id": 1,
      "explanation": "...",
      "neighbors": ["A", "B"]
    }
  ]
}
`

  // ---- Call Gemini ----
  const { text } = await generateText({
    model: google("gemini-3-flash-preview"),
    prompt
  })

  let parsed
  try {
    parsed = JSON.parse(text)
  } catch (e) {
    return { error: "Failed to parse LLM response", raw: text }
  }

  const results: any[] = []

  // ---- Evaluate Each Proposal ----
  for (const proposal of parsed.proposals) {
    const forecast = await $fetch<ForecastResponse>("http://localhost:8000/forecast", {
      method: "POST",
      body: {
        target,
        neighbors: proposal.neighbors,
        horizon
      }
    })

    results.push({
      ...proposal,
      mae: forecast.mae,
      predicted_return: forecast.predicted_return
    })
  }

  // ---- Pick Best ----
  const best = results.reduce((bestSoFar, current) => {
    if (!bestSoFar) return current
    return current.mae < bestSoFar.mae ? current : bestSoFar
  }, null)

  return {
    target,
    horizon,
    proposals: results,
    best
  }
})