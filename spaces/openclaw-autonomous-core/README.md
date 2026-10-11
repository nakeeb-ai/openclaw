# OpenClaw-Autonomous-Core (Hugging Face Space)

This Space scaffold runs an autonomous multi-model router for OpenClaw-style agent workflows.

## Features

- Default brain: `THUDM/glm-5.1`
- Universal router entrypoint: `model_router.generate(prompt, task="general")`
- Task-aware dynamic routing (`code`, `long_reasoning`, `fast_response`, `multilingual`, `embedding`)
- Auto-discovery from `https://huggingface.co/api/models`
- Model registry persistence in `/memory/model_registry.json`
- 12-hour registry refresh + daily self-improvement ranking
- Retry/failover + per-model latency/reliability tracking
- Prompt response cache

## Environment Variables

- `HF_TOKEN` (recommended for private access and higher rate limits)

## Hugging Face Space Setup

1. Create a **private** Space named **OpenClaw-Autonomous-Core**.
2. Set SDK to **Gradio** and Python runtime.
3. Upload this folder (`spaces/openclaw-autonomous-core`) as the Space root.
4. Add `HF_TOKEN` as a secret.
5. Start the Space.

On startup, the app prints:

`OpenClaw Multi-Model Autonomous System Active`

## OpenClaw Integration

Route all OpenClaw model decisions through:

```python
from model_router import openclaw_generate

response = openclaw_generate("Plan an autonomous deployment", task="long_reasoning")
```

Replace existing static model invocations with `openclaw_generate(...)` or `generate(...)`.
