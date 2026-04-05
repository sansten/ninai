import { parseSseChunk } from "../src/utils/stream.js";

const payload = "data: {\"event\":\"heartbeat\",\"data\":{\"ok\":true}}\n\n";
const events = parseSseChunk(payload);

if (events.length !== 1) {
  throw new Error("expected one parsed event");
}
