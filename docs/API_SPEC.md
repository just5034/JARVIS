# JARVIS API Specification

JARVIS exposes an OpenAI-compatible API so that any client built for OpenAI (including GRACE) can use JARVIS as a drop-in replacement.

---

## Base URL

```
http://localhost:8000/v1
```

No authentication required for local deployment. API key field is accepted but ignored.

---

## Endpoints

### POST /v1/chat/completions

The primary inference endpoint. OpenAI-compatible.

**Request:**
```json
{
  "model": "",
  "messages": [
    {"role": "system", "content": "You are a particle physics expert."},
    {"role": "user", "content": "Calculate the Higgs boson decay width to b-bbar at leading order."}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "top_p": 0.95,
  "stream": false,
  "stop": ["\n\n---"],
  "n": 1
}
```

**Model field behavior:**
| Value | Behavior |
|-------|----------|
| `""` or omitted | Router auto-selects domain and brain |
| `"math"` | Force math brain |
| `"physics"` | Force physics brain |
| `"code"` | Force code brain |
| `"chemistry"` | Force ChemLLM specialist |
| `"biology"` | Force BioMistral specialist |
| `"protein"` | Force ESM3 specialist |
| `"genomics"` | Force Evo 2 specialist |
| `"auto"` | Same as empty — router decides |

**Response (non-streaming):**
```json
{
  "id": "jarvis-abc123",
  "object": "chat.completion",
  "created": 1711234567,
  "model": "physics:qwen-32b-physics-lora",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The leading-order partial width for H → bb̄ is given by..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 512,
    "total_tokens": 554
  },
  "jarvis_metadata": {
    "routed_domain": "physics",
    "routed_difficulty": "medium",
    "inference_strategy": "best_of_4",
    "active_adapter": "physics_hep",
    "candidates_generated": 4,
    "verification_score": 0.92,
    "rag_passages_retrieved": 2,
    "time_ms": 8432
  }
}
```

**Note:** `jarvis_metadata` is a non-standard extension. Clients that don't expect it will ignore it. GRACE can use it for logging and debugging.

**Response (streaming):**
```
data: {"id":"jarvis-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"jarvis-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"jarvis-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" leading"},"finish_reason":null}]}

...

data: {"id":"jarvis-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Important:** Streaming is only available for easy/medium queries (single pass or best-of-N with early winner). Hard queries with verification cannot stream until the best candidate is selected.

---

### GET /v1/models

List available models and brains.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "auto", "object": "model", "owned_by": "jarvis", "description": "Auto-routed (default)"},
    {"id": "math", "object": "model", "owned_by": "jarvis", "description": "Math brain (R1-Distill-70B)"},
    {"id": "physics", "object": "model", "owned_by": "jarvis", "description": "Physics brain (Qwen-32B + physics LoRA)"},
    {"id": "code", "object": "model", "owned_by": "jarvis", "description": "Code brain (Qwen-32B + code LoRA)"},
    {"id": "chemistry", "object": "model", "owned_by": "jarvis", "description": "ChemLLM-7B (on-demand)"},
    {"id": "biology", "object": "model", "owned_by": "jarvis", "description": "BioMistral-7B (on-demand)"},
    {"id": "protein", "object": "model", "owned_by": "jarvis", "description": "ESM3-open 1.4B (on-demand)"},
    {"id": "genomics", "object": "model", "owned_by": "jarvis", "description": "Evo 2 7B (on-demand)"}
  ]
}
```

---

### GET /health

System health and status.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 86400,
  "memory": {
    "total_gb": 128.0,
    "used_gb": 46.2,
    "available_gb": 81.8,
    "loaded_models": [
      {"name": "qwen-32b-base", "size_gb": 16.0, "status": "resident"},
      {"name": "physics_hep_lora", "size_gb": 0.3, "status": "active_adapter"},
      {"name": "think_prm", "size_gb": 0.8, "status": "resident"},
      {"name": "draft_model", "size_gb": 0.8, "status": "resident"},
      {"name": "router", "size_gb": 0.06, "status": "resident"},
      {"name": "rag_index", "size_gb": 5.0, "status": "resident"}
    ]
  },
  "gpu": {
    "temperature_c": 62,
    "utilization_pct": 45,
    "power_w": 87
  },
  "inference": {
    "requests_total": 1542,
    "requests_active": 2,
    "avg_latency_ms": 3200,
    "model_swaps_total": 23
  }
}
```

---

### POST /admin/load

Manually load or unload a model. For debugging and testing.

**Request:**
```json
{
  "action": "load",
  "model": "chemistry"
}
```

**Response:**
```json
{
  "status": "loaded",
  "model": "chemistry",
  "size_gb": 3.5,
  "load_time_ms": 7200,
  "memory_used_gb": 49.7,
  "memory_available_gb": 78.3
}
```

---

### GET /admin/memory

Detailed memory breakdown.

**Response:**
```json
{
  "total_gb": 128.0,
  "os_overhead_gb": 10.0,
  "framework_overhead_gb": 7.0,
  "models": {
    "resident": [
      {"name": "qwen-32b-base", "size_gb": 16.0, "type": "base_model"},
      {"name": "think_prm", "size_gb": 0.8, "type": "verifier"},
      {"name": "draft_model", "size_gb": 0.8, "type": "speculative"},
      {"name": "rag_index", "size_gb": 5.0, "type": "index"},
      {"name": "router", "size_gb": 0.06, "type": "classifier"}
    ],
    "active_adapter": {"name": "physics_hep", "size_gb": 0.3},
    "cached_specialists": [
      {"name": "chemllm-7b", "size_gb": 3.5, "last_used": "2026-03-23T14:30:00Z"}
    ],
    "available_adapters": ["physics_general", "physics_hep", "code_general", "code_hep", "math"]
  },
  "used_gb": 43.46,
  "available_gb": 84.54,
  "safety_margin_gb": 5.0
}
```

---

## Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Model 'chemistry' failed to load: insufficient memory (need 3.5 GB, have 2.1 GB available)",
    "type": "insufficient_memory",
    "code": 503
  }
}
```

| Code | Type | When |
|------|------|------|
| 400 | `invalid_request` | Malformed request body |
| 404 | `model_not_found` | Requested model/specialist not in registry |
| 408 | `timeout` | Inference exceeded timeout for difficulty level |
| 503 | `insufficient_memory` | Cannot load requested model within memory budget |
| 503 | `model_load_failed` | Model weights corrupted or missing |
| 504 | `inference_timeout` | Model loaded but inference timed out |

---

## Rate Limits

No rate limits for local deployment. The DGX Spark hardware is the natural bottleneck. If multiple GRACE instances connect, requests are queued and processed in order. vLLM's continuous batching handles concurrent requests to the same model efficiently.

---

## Non-Text Specialist API Adapters

For specialist models that don't operate on natural language (ESM3, Evo 2), JARVIS provides transparent translation:

**Protein queries (ESM3):**
- User sends: `"Predict the structure of sequence MVLSPADKTNVKAAW..."`
- Router detects protein domain
- JARVIS extracts the amino acid sequence from the message
- ESM3 runs structure prediction
- Response wraps the output in a natural language explanation with structured data

**DNA queries (Evo 2):**
- User sends: `"What is the likely effect of mutation G>A at position 12345 in BRCA1?"`
- Router detects genomics domain
- JARVIS formats the query for Evo 2's variant effect prediction
- Response wraps the prediction in natural language

These adapters live in `src/specialists/adapters/` and implement a common interface:
```python
class SpecialistAdapter:
    def parse_input(self, messages: list[dict]) -> ModelInput: ...
    def format_output(self, model_output: Any) -> str: ...
```
