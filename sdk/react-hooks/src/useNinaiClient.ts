import { useMemo } from "react";
import { NinaiClient, type NinaiClientOptions } from "@ninai/sdk";

export function useNinaiClient(options: NinaiClientOptions): NinaiClient {
  return useMemo(() => new NinaiClient(options), [
    options.apiKey,
    options.baseUrl,
    options.orgId,
    options.maxRetries,
    options.timeout,
  ]);
}
