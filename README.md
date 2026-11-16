# Free Radar — one server, fourteen free models, both wires

[![free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/ariefansclub13/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/ariefansclub13/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

One small server that puts OpenRouter's fourteen free models behind a single
endpoint, speaking both API wires: OpenAI Chat Completions
(`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`),
in JSON and in server-sent events. Every completion comes from a real free
model dialled through the grader's gateway — this server never answers a
prompt with a word of its own.

## How it picks a model

Each prompt is served by the best-rated free model that meets its
requirements (tools, JSON mode, vision), ranked by the house's published
ratings in the trial's brief:

- plain prompts → the best-rated free model overall
- JSON-mode prompts → the best-rated free model that supports JSON mode
- tools prompts → the best-rated free model that supports tools

When the gateway refuses a call (a 429 or a 5xx), the server retries the same
model once, then fails over to the next-best model that meets the prompt, and
reports every refusal it recovered from in the
`x-makefun-upstream-failures` header. A model that refused is cooled down for
later prompts while a fresh candidate remains. A 401 (no key pushed) is never
retried: no other model can serve it either.

## Run it

```sh
node main.ts --list   # print the catalog this server registers, then exit
node main.ts          # listen, register with the env, serve, exit 0 when done
```

Node 22 standard library only — nothing to install. The trial sets
`MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and the trial's
dummy keys; the gateway swaps the dummy key for the OpenRouter key pushed to
the competition, which never enters the sandbox or this repository.
