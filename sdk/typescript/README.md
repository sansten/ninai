# @ninai/sdk

TypeScript SDK for Ninai Cognitive OS.

## Quick Start

```ts
import { NinaiClient } from "@ninai/sdk";

const client = new NinaiClient({
  apiKey: process.env.NINAI_API_KEY!,
  baseUrl: "http://localhost:8000/api/v1"
});

const result = await client.cognitive.decide("Summarize current incident status");
console.log(result);
```
