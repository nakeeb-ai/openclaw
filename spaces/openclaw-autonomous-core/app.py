from __future__ import annotations

import gradio as gr

from model_router import MODEL_DEFAULT, ROUTER, generate


TASK_CHOICES = [
    "general",
    "chat",
    "text-generation",
    "summarization",
    "translation",
    "code",
    "code-generation",
    "long_reasoning",
    "fast_response",
    "multilingual",
    "embedding",
]


def run_router(prompt: str, task: str) -> str:
    if not prompt.strip():
        return "Please provide a prompt."
    return generate(prompt=prompt, task=task)


def registry_snapshot() -> str:
    ROUTER.maybe_refresh_registry()
    return ROUTER.memory_path.read_text(encoding="utf-8")


with gr.Blocks(title="OpenClaw Autonomous Core") as demo:
    gr.Markdown("# OpenClaw-Autonomous-Core")
    gr.Markdown(f"**Default brain:** `{MODEL_DEFAULT}`")

    prompt = gr.Textbox(label="Prompt", lines=8, placeholder="Ask anything...")
    task = gr.Dropdown(TASK_CHOICES, value="general", label="Task")

    with gr.Row():
        run_btn = gr.Button("Generate")
        refresh_btn = gr.Button("Refresh Registry")

    output = gr.Textbox(label="Output", lines=12)
    registry = gr.Code(label="/memory/model_registry.json", language="json")

    run_btn.click(run_router, inputs=[prompt, task], outputs=output)
    refresh_btn.click(registry_snapshot, outputs=registry)

if __name__ == "__main__":
    print("OpenClaw Multi-Model Autonomous System Active")
    demo.launch()
