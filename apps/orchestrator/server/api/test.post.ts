import { defineEventHandler, readBody } from "h3"

export default defineEventHandler(async (event) => {
    const body = await readBody(event)

    const response = await $fetch("http://localhost:8000/forecast", {
        method: "POST",
        body: {
        target: "INFY",
        neighbors: ["TCS"],
        horizon: 5
        }
    })

    return response
})