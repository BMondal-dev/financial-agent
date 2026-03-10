import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';

export const anannas = createOpenAICompatible({
    name: 'anannas',
    apiKey: process.env.ANANNAS_API_KEY,
    baseURL: 'https://unoffending-unconfinable-augustine.ngrok-free.dev/v1/',
});
  
export const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY,
    extraBody: {
        response_format: { type: "json_object" }
    }
});  
  
