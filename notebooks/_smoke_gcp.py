import json, urllib.request, sys

OLLAMA_URL = 'http://localhost:11435'
LLM_MODEL  = 'qwen2.5:1.5b'

def _ollama_generate(prompt, model=LLM_MODEL, timeout=30, num_ctx=512):
    payload = json.dumps({
        'model': model, 'prompt': prompt, 'stream': False,
        'options': {'num_ctx': num_ctx, 'temperature': 0, 'top_p': 1.0},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL.rstrip('/') + '/api/generate',
        data=payload, headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())['response'].strip()
    except Exception as e:
        return f'ERROR: {e}'

context = """[Caroline] I went to the grocery store yesterday.
[Melanie] Oh nice! Did you get anything fun?
[Caroline] Just basics. But I did pick up ingredients for my grandmother's famous apple pie recipe.
[Melanie] That sounds amazing. Is your grandmother still baking?
[Caroline] She passed away last year. I make her recipes to remember her."""

prompt = (
    "Answer the question using only information from the conversation below.\n"
    "Be concise. If not in the conversation, say: Not mentioned.\n\n"
    f"CONVERSATION:\n{context}\n\n"
    "QUESTION: What recipe is Caroline making?\nANSWER:"
)

ans = _ollama_generate(prompt)
passed = 'apple pie' in ans.lower()
print(f'Ollama URL:  {OLLAMA_URL}')
print(f'Model:       {LLM_MODEL}')
print(f'Answer:      {ans!r}')
print(f'Pass:        {"YES" if passed else "FAIL"}')
sys.exit(0 if passed else 1)
