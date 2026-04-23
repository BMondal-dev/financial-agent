import { readFile } from "fs/promises"
import { defineEventHandler, readBody } from "h3"
import { join } from "path"

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  const { target, horizon = 5 } = body

  if (!target) {
    return { error: "target is required" }
  }

  // ---- Load metadata.json ----
  const metadataPath = join(process.cwd(), "../../data/metadata.json")
  const metadataRaw = await readFile(metadataPath, "utf-8")
  const metadata = JSON.parse(metadataRaw)

  if (!metadata[target]) {
    return { error: `Target ${target} not found in metadata` }
  }

  // ---- Baseline (no neighbors) ----
  const baseline = await $fetch("http://localhost:8000/run-forecast", {
    method: "POST",
    body: {
      target,
      neighbors: [],
      horizon,
      source: "run-experiment",
      rationale: "baseline: no neighbors"
    }
  })

  // ---- Correlated (top 3) ----
  const correlatedNeighbors =
    metadata[target].top_correlated
      ?.slice(0, 3)
      .map((n: any) => n.symbol) || []

  const correlated = await $fetch("http://localhost:8000/run-forecast", {
    method: "POST",
    body: {
      target,
      neighbors: correlatedNeighbors,
      horizon,
      source: "run-experiment",
      rationale: "top correlated (metadata)"
    }
  })

  // ---- Sector (first 3 same sector) ----
  const sectorNeighbors =
    metadata[target].same_sector
      ?.slice(0, 3) || []

  const sector = await $fetch("http://localhost:8000/run-forecast", {
    method: "POST",
    body: {
      target,
      neighbors: sectorNeighbors,
      horizon,
      source: "run-experiment",
      rationale: "same sector (metadata)"
    }
  })

  // ---- Determine Best (ranking prefers models that beat zero baseline) ----
  const results = {
    baseline,
    correlated,
    sector
  }

  const rank = (v: any) => v?.mae_for_ranking ?? v?.mae ?? Number.POSITIVE_INFINITY

  const best = Object.entries(results).reduce((bestSoFar, current) => {
    const [name, value]: any = current
    if (value?.error || typeof value?.mae !== "number") return bestSoFar

    if (!bestSoFar) return { name, mae: value.mae, mae_for_ranking: rank(value) }

    return rank(value) < bestSoFar.mae_for_ranking
      ? { name, mae: value.mae, mae_for_ranking: rank(value) }
      : bestSoFar
  }, null as any)

  return {
    target,
    horizon,
    results,
    best: best?.name || null
  }
})