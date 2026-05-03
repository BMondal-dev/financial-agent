import { generateText, Output } from "ai"
import { google } from "@ai-sdk/google"
import { z } from "zod"

const FASTAPI = "http://localhost:8000"
const AGENT_SYSTEM_PROMPT =
  "You are a quantitative finance research agent. Return strictly valid JSON that matches the schema."

export const requestSchema = z.object({
  target: z.string().min(1),
  horizon: z.number().int().positive().max(30).optional(),
})

export type ForecastLike = {
  mae?: number
  mae_for_ranking?: number
  error?: string
}

export type ExperimentProposal = {
  experiment_id: string
  rationale: string
  neighbors: string[]
}

export type ExperimentResult = ExperimentProposal & {
  mae?: number
  mae_for_ranking?: number
  error?: string
  [key: string]: any
}

export const singleRoundExperimentSchema = z.object({
  explanation: z.string().min(1),
  neighbors: z.array(z.string().min(1)).min(1).max(3),
})

export const multiRoundExperimentSchema = z.object({
  experiment_id: z.string().min(1),
  rationale: z.string().min(1),
  neighbors: z.array(z.string().min(1)).min(1).max(3),
})

export function rankingMae(r: ForecastLike): number {
  return r.mae_for_ranking ?? r.mae ?? Number.POSITIVE_INFINITY
}

function cleanNeighbors(neighbors: string[], target: string): string[] {
  const unique = [...new Set(neighbors.map((n) => n.trim().toUpperCase()))]
  return unique.filter((n) => n && n !== target).slice(0, 3)
}

function normalizeProposal(
  proposal: ExperimentProposal,
  target: string,
  fallbackId: string,
): ExperimentProposal {
  const normalizedNeighbors = cleanNeighbors(proposal.neighbors, target)
  return {
    experiment_id: proposal.experiment_id || fallbackId,
    rationale: proposal.rationale || "No rationale provided.",
    neighbors: normalizedNeighbors,
  }
}

export async function curator(target: string, horizon: number) {
  const [metadata, candidates] = await Promise.all([
    $fetch(`${FASTAPI}/metadata/${target}`),
    $fetch(`${FASTAPI}/candidate-neighbors/${target}`),
  ])

  let graphNeighbors: any[] = []
  try {
    graphNeighbors = await $fetch(`${FASTAPI}/graph-neighbors/${target}`)
  } catch {
    graphNeighbors = []
  }

  let pastExperiments: any[] = []
  try {
    pastExperiments = await $fetch(
      `${FASTAPI}/best-neighbors/${target}?horizon=${horizon}`,
    )
  } catch {
    pastExperiments = []
  }

  return { metadata, candidates, graphNeighbors, pastExperiments }
}

export async function plannerSingleRound(
  target: string,
  context: Awaited<ReturnType<typeof curator>>,
): Promise<Array<{ explanation: string; neighbors: string[] }>> {
  const { output } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.object({
      schema: z.object({
        experiments: z.array(singleRoundExperimentSchema).length(3),
      }),
    }),
    system: AGENT_SYSTEM_PROMPT,
    prompt: `
Target stock: ${target}
Metadata: ${JSON.stringify(context.metadata)}
Candidate neighbors: ${JSON.stringify(context.candidates)}
Past successful experiments: ${JSON.stringify(context.pastExperiments)}

Generate exactly 3 diverse neighbor hypotheses.
Each hypothesis must include:
- explanation
- neighbors (1 to 3 tickers)
`,
  })

  return output.experiments.map((exp) => ({
    explanation: exp.explanation,
    neighbors: cleanNeighbors(exp.neighbors, target),
  }))
}

export async function plannerRound(
  target: string,
  round: number,
  prompt: string,
): Promise<ExperimentProposal[]> {
  const { output } = await generateText({
    model: google("gemini-flash-latest"),
    output: Output.array({ element: multiRoundExperimentSchema }),
    system: AGENT_SYSTEM_PROMPT,
    prompt,
  })

  return output.map((proposal, index) =>
    normalizeProposal(proposal, target, `r${round}_exp${index + 1}`),
  )
}

export async function runForecast(
  target: string,
  neighbors: string[],
  horizon: number,
  meta: { source: string; experiment_id?: string; rationale?: string; orchestrator_round?: number },
) {
  return await $fetch(`${FASTAPI}/run-forecast`, {
    method: "POST",
    body: {
      target,
      neighbors,
      horizon,
      source: meta.source,
      experiment_id: meta.experiment_id,
      rationale: meta.rationale,
      orchestrator_round: meta.orchestrator_round,
    },
  })
}

export async function runner(
  target: string,
  horizon: number,
  source: string,
  proposals: ExperimentProposal[],
  orchestratorRound?: number,
): Promise<ExperimentResult[]> {
  const results: ExperimentResult[] = []

  for (const proposal of proposals) {
    try {
      const forecast: any = await runForecast(target, proposal.neighbors, horizon, {
        source,
        experiment_id: proposal.experiment_id,
        rationale: proposal.rationale,
        orchestrator_round: orchestratorRound,
      })
      results.push({
        experiment_id: proposal.experiment_id,
        rationale: proposal.rationale,
        neighbors: proposal.neighbors,
        ...forecast,
      })
    } catch (e) {
      results.push({
        experiment_id: proposal.experiment_id,
        rationale: proposal.rationale,
        neighbors: proposal.neighbors,
        mae: Number.POSITIVE_INFINITY,
        error: String(e),
      })
    }
  }

  return results
}

export function pickBest(results: ExperimentResult[]): ExperimentResult | null {
  const valid = results
    .filter((r) => !r.error && typeof r.mae === "number" && Number.isFinite(r.mae))
    .sort((a, b) => rankingMae(a) - rankingMae(b))
  return valid[0] ?? null
}

export function reporterSingle(
  target: string,
  horizon: number,
  experiments: ExperimentResult[],
) {
  return {
    target,
    horizon,
    experiments,
    best: pickBest(experiments),
  }
}

export function reporterMultiRound(
  target: string,
  horizon: number,
  rounds: { round1: ExperimentResult[]; round2: ExperimentResult[]; round3: ExperimentResult[] },
) {
  const merged = [...rounds.round1, ...rounds.round2, ...rounds.round3]
  return {
    target,
    horizon,
    round1: rounds.round1,
    round2: rounds.round2,
    round3: rounds.round3,
    best: pickBest(merged),
  }
}
