import { defineEventHandler, readBody } from "h3"

const FASTAPI = "http://localhost:8000"

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const target = body?.target
  const neighbors = Array.isArray(body?.neighbors) ? body.neighbors : []
  const horizon = Number(body?.horizon || 5)

  if (!target) {
    return { error: "target is required" }
  }

  try {
    const benchmark = await $fetch(`${FASTAPI}/compare-models`, {
      method: "POST",
      body: { target, neighbors, horizon }
    })

    return benchmark
  } catch (e) {
    return {
      target,
      horizon,
      neighbors,
      error: String(e)
    }
  }
})
