# Agent evaluation — agentic vs naive RAG

- Generated: `2026-10-15T08:28:18+00:00`
- Gold questions: **35**

## Backend `transformers` (device `cpu`)

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| agentic | 0.700 | 0.714 | 0.986 | 1.000 | 3.343 | 21.097 |
| naive | 0.767 | 0.586 | 0.000 | 0.000 | 0.000 | 6.821 |

Per category (agentic):

| Category | n | Answer match | Citation precision | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| lookup | 10 | 0.400 | 0.650 | 3.000 | 23.906 |
| multi_hop | 7 | 1.000 | 0.857 | 3.286 | 16.784 |
| numeric | 8 | 0.750 | 0.375 | 4.000 | 25.081 |
| refusal | 5 | n/a | 1.000 | 3.000 | 17.599 |
| supersede | 5 | 0.800 | 0.900 | 3.400 | 18.640 |

## Backend `extractive` (device `cpu`)

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| agentic | 0.933 | 0.795 | 0.986 | 1.000 | 3.286 | 0.004 |
| naive | 0.867 | 0.610 | 0.000 | 0.200 | 0.000 | 0.001 |

Per category (agentic):

| Category | n | Answer match | Citation precision | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| lookup | 10 | 0.900 | 0.750 | 2.900 | 0.004 |
| multi_hop | 7 | 0.857 | 0.762 | 3.286 | 0.003 |
| numeric | 8 | 1.000 | 0.625 | 4.000 | 0.004 |
| refusal | 5 | n/a | 1.000 | 2.800 | 0.005 |
| supersede | 5 | 1.000 | 1.000 | 3.400 | 0.003 |

