# @ninai/react-hooks

React hooks for the Ninai TypeScript SDK.

## Example

```tsx
import { useNinaiClient, useCognitiveRead } from "@ninai/react-hooks";

function SearchWidget() {
  const client = useNinaiClient({ apiKey: "...", baseUrl: "http://localhost:8000/api/v1" });
  const { run, loading, data } = useCognitiveRead(client);

  return (
    <button disabled={loading} onClick={() => void run("recent incidents")}>Search</button>
  );
}
```
