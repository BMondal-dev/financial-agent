import { defineEventHandler, readBody } from "h3"
import { generateText, Output } from "ai"
import { google } from "@ai-sdk/google"
import { z } from "zod"

const FASTAPI = "http://localhost:8000"

function rankingMae(r: { mae_for_ranking?: number; mae?: number }) {
  return r.mae_for_ranking ?? r.mae ?? Number.POSITIVE_INFINITY
}

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  const target = body.target
  const horizon = body.horizon || 5

  // 1️⃣ Fetch Metadata
  // Using native fetch or $fetch (Nuxt/Nitro)
  const metadata = await $fetch(`${FASTAPI}/metadata/${target}`)
  const candidates = await $fetch(`${FASTAPI}/candidate-neighbors/${target}`)
  
  // Fetch past experiments for memory
  const pastExperiments = await $fetch(
    `${FASTAPI}/best-neighbors/${target}?horizon=${horizon}`
  )

  // 2️⃣ Ask LLM for experiments
  // We use generateObject here because you want a typed JSON response
  const { output  } = await generateText({
    model: google("gemini-flash-lite-latest"), 
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
      Metadata: ${JSON.stringify(metadata, null, 2)}
      Candidate neighbors: ${JSON.stringify(candidates)}

      Propose 3 different neighbor sets (3 stocks each) to forecast the target.
      Strategies can include: sector similarity, correlation similarity, 
      volatility regime similarity, consumption proxies, and cross-sector signals.
      Return diverse hypotheses.
    `,
  })

  const proposals = output.experiments
  const results: any[] = []

  console.log(proposals);

  // 3️⃣ Run experiments
  for (const exp of proposals) {
    const forecast: any = await $fetch(`${FASTAPI}/run-forecast`, {
      method: "POST",
      body: {
        target,
        neighbors: exp.neighbors,
        horizon,
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

  return {
    target,
    horizon,
    experiments: results,
    best
  }
})