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
  const baseline = await $fetch("http://localhost:8000/forecast", {
    method: "POST",
    body: {
      target,
      neighbors: [],
      horizon
    }
  })

  // ---- Correlated (top 3) ----
  const correlatedNeighbors =
    metadata[target].top_correlated
      ?.slice(0, 3)
      .map((n: any) => n.symbol) || []

  const correlated = await $fetch("http://localhost:8000/forecast", {
    method: "POST",
    body: {
      target,
      neighbors: correlatedNeighbors,
      horizon
    }
  })

  // ---- Sector (first 3 same sector) ----
  const sectorNeighbors =
    metadata[target].same_sector
      ?.slice(0, 3) || []

  const sector = await $fetch("http://localhost:8000/forecast", {
    method: "POST",
    body: {
      target,
      neighbors: sectorNeighbors,
      horizon
    }
  })

  // ---- Determine Best ----
  const results = {
    baseline,
    correlated,
    sector
  }

  const best = Object.entries(results).reduce((bestSoFar, current) => {
    const [name, value]: any = current
    if (!value?.mae) return bestSoFar

    if (!bestSoFar) return { name, mae: value.mae }

    return value.mae < bestSoFar.mae
      ? { name, mae: value.mae }
      : bestSoFar
  }, null as any)

  return {
    target,
    horizon,
    results,
    best: best?.name || null
  }
})