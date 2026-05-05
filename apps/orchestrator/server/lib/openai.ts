import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';

export const chutes = createOpenAICompatible({
    name: 'chutes',
    apiKey: process.env.CHUTES_API_KEY,
    baseURL: 'https://llm.chutes.ai/v1',
});

export const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY,
    extraBody: {
        response_format: { type: "json_object" }
    }
});

