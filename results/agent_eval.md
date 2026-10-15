# Agent evaluation — agentic vs naive RAG

- Generated: `2026-10-15T07:45:11+00:00`
- Gold questions: **35**

## Backend `transformers` (device `cpu`)

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| agentic | 0.700 | 0.571 | 0.986 | 1.000 | 3.343 | 11.994 |
| naive | 0.767 | 0.414 | 0.000 | 0.000 | 0.000 | 4.104 |

Per category (agentic):

| Category | n | Answer match | Citation precision | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| lookup | 10 | 0.400 | 0.450 | 3.000 | 14.829 |
| multi_hop | 7 | 1.000 | 0.714 | 3.286 | 7.939 |
| numeric | 8 | 0.750 | 0.125 | 4.000 | 13.059 |
| refusal | 5 | n/a | 1.000 | 3.000 | 8.642 |
| supersede | 5 | 0.800 | 0.900 | 3.400 | 13.647 |

## Backend `extractive` (device `cpu`)

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| agentic | 0.933 | 0.795 | 0.986 | 1.000 | 3.286 | 0.002 |
| naive | 0.867 | 0.610 | 0.000 | 0.200 | 0.000 | 0.001 |

Per category (agentic):

| Category | n | Answer match | Citation precision | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| lookup | 10 | 0.900 | 0.750 | 2.900 | 0.001 |
| multi_hop | 7 | 0.857 | 0.762 | 3.286 | 0.001 |
| numeric | 8 | 1.000 | 0.625 | 4.000 | 0.002 |
| refusal | 5 | n/a | 1.000 | 2.800 | 0.002 |
| supersede | 5 | 1.000 | 1.000 | 3.400 | 0.002 |

