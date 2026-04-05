import { NinaiClient } from "../src/client.js";

const client = new NinaiClient({ apiKey: "test-key", baseUrl: "http://localhost:8000/api/v1" });

if (!client.cognitive || !client.memory || !client.events) {
  throw new Error("client resources not initialized");
}
