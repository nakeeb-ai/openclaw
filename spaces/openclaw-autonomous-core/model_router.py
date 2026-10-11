from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import InferenceClient

MODEL_DEFAULT = "THUDM/glm-5.1"
HF_MODELS_API = "https://huggingface.co/api/models"
REFRESH_INTERVAL_SECONDS = 12 * 60 * 60
IMPROVEMENT_INTERVAL_SECONDS = 24 * 60 * 60
MAX_RETRIES = 2

TASK_TO_PIPELINE = {
    "general": "text-generation",
    "text-generation": "text-generation",
    "chat": "text-generation",
    "summarization": "summarization",
    "translation": "translation",
    "code": "text-generation",
    "code-generation": "text-generation",
    "long_reasoning": "text-generation",
    "fast_response": "text-generation",
    "multilingual": "translation",
    "embedding": "feature-extraction",
    "embeddings": "feature-extraction",
}

TASK_BACKUPS = {
    "code": ["deepseek-ai/deepseek-coder-6.7b-instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
    "long_reasoning": [MODEL_DEFAULT, "Qwen/Qwen3-32B"],
    "fast_response": ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-3B-Instruct"],
    "multilingual": ["facebook/nllb-200-distilled-600M", "Helsinki-NLP/opus-mt-en-fr"],
    "embedding": ["sentence-transformers/all-MiniLM-L6-v2"],
    "embeddings": ["sentence-transformers/all-MiniLM-L6-v2"],
    "general": [MODEL_DEFAULT, "Qwen/Qwen2.5-14B-Instruct"],
}


@dataclass
class ModelStats:
    calls: int = 0
    failures: int = 0
    latency_ms_total: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0

    @property
    def reliability(self) -> float:
        if self.calls == 0:
            return 1.0
        return max(0.0, 1.0 - (self.failures / self.calls))


@dataclass
class RouterMemory:
    updated_at: float = 0.0
    improved_at: float = 0.0
    discovered_models: dict[str, list[str]] = field(default_factory=dict)
    model_stats: dict[str, ModelStats] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        stats = {name: asdict(stat) for name, stat in self.model_stats.items()}
        return {
            "updated_at": self.updated_at,
            "improved_at": self.improved_at,
            "discovered_models": self.discovered_models,
            "model_stats": stats,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RouterMemory":
        stats_payload = payload.get("model_stats", {})
        stats = {
            name: ModelStats(
                calls=int(values.get("calls", 0)),
                failures=int(values.get("failures", 0)),
                latency_ms_total=float(values.get("latency_ms_total", 0.0)),
            )
            for name, values in stats_payload.items()
        }
        return cls(
            updated_at=float(payload.get("updated_at", 0.0)),
            improved_at=float(payload.get("improved_at", 0.0)),
            discovered_models={k: list(v) for k, v in payload.get("discovered_models", {}).items()},
            model_stats=stats,
        )


class ModelRouter:
    def __init__(
        self,
        token: str | None = None,
        memory_file: str = "/memory/model_registry.json",
        timeout_seconds: int = 90,
    ) -> None:
        self.token = token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.token, timeout=timeout_seconds)
        self.timeout_seconds = timeout_seconds
        self.cache: dict[str, str] = {}
        self.memory_path = self._resolve_memory_path(memory_file)
        self.memory = self._load_memory()

    def _resolve_memory_path(self, configured_path: str) -> Path:
        preferred = Path(configured_path)
        try:
            preferred.parent.mkdir(parents=True, exist_ok=True)
            return preferred
        except OSError:
            fallback = Path(__file__).resolve().parent / "memory" / "model_registry.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            return fallback

    def _load_memory(self) -> RouterMemory:
        if self.memory_path.exists():
            try:
                return RouterMemory.from_json(json.loads(self.memory_path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass
        return RouterMemory()

    def _save_memory(self) -> None:
        self.memory_path.write_text(json.dumps(self.memory.to_json(), indent=2), encoding="utf-8")

    def maybe_refresh_registry(self) -> None:
        now = time.time()
        if now - self.memory.updated_at < REFRESH_INTERVAL_SECONDS:
            return

        discovered = self._discover_models()
        if discovered:
            self.memory.discovered_models = discovered
            self.memory.updated_at = now
            self._save_memory()

    def maybe_improve_router(self) -> None:
        now = time.time()
        if now - self.memory.improved_at < IMPROVEMENT_INTERVAL_SECONDS:
            return

        for task, models in list(self.memory.discovered_models.items()):
            ranked = sorted(models, key=self._score_model, reverse=True)
            self.memory.discovered_models[task] = ranked

        self.memory.improved_at = now
        self._save_memory()

    def _score_model(self, model: str) -> float:
        stat = self.memory.model_stats.get(model, ModelStats())
        # Higher score = better model for future routing. Prioritize reliability,
        # then latency (lower is better) and total usage confidence.
        usage_bonus = min(stat.calls, 100) / 1000
        latency_penalty = stat.avg_latency_ms / 5000 if stat.avg_latency_ms else 0
        return (stat.reliability * 2.0) + usage_bonus - latency_penalty

    def _discover_models(self) -> dict[str, list[str]]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        task_queries = {
            "text-generation": "text-generation",
            "summarization": "summarization",
            "translation": "translation",
            "feature-extraction": "sentence-transformers",
        }
        discovered: dict[str, list[str]] = {}

        for task, search in task_queries.items():
            params = {
                "search": search,
                "sort": "downloads",
                "direction": "-1",
                "limit": 25,
                "full": "true",
            }
            try:
                response = requests.get(
                    HF_MODELS_API,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException:
                continue

            ranked: list[tuple[float, str]] = []
            now = time.time()
            for item in payload:
                model_id = item.get("id")
                if not model_id:
                    continue
                downloads = float(item.get("downloads") or 0)
                likes = float(item.get("likes") or 0)
                last_modified = item.get("lastModified") or ""
                recency_boost = 0.0
                if isinstance(last_modified, str) and len(last_modified) >= 10:
                    # Coarse recency signal: give a small boost if updated this year.
                    if str(time.gmtime(now).tm_year) in last_modified:
                        recency_boost = 200.0
                score = downloads + (likes * 10.0) + recency_boost
                ranked.append((score, model_id))

            ranked.sort(reverse=True)
            discovered[task] = [model for _, model in ranked[:8]]

        discovered.setdefault("text-generation", [])
        if MODEL_DEFAULT not in discovered["text-generation"]:
            discovered["text-generation"].insert(0, MODEL_DEFAULT)

        return discovered

    def select_best_model(self, task: str, prompt: str | None = None) -> str:
        task_key = (task or "general").strip().lower()
        pipeline = TASK_TO_PIPELINE.get(task_key, "text-generation")
        prompt_len = len(prompt or "")

        candidates = list(TASK_BACKUPS.get(task_key, []))
        discovered = self.memory.discovered_models.get(pipeline, [])
        candidates.extend(discovered)

        if task_key == "fast_response" or prompt_len < 500:
            candidates.extend(TASK_BACKUPS["fast_response"])
        if task_key in {"long_reasoning", "general"} or prompt_len > 4000:
            candidates.extend(TASK_BACKUPS["long_reasoning"])

        candidates.append(MODEL_DEFAULT)

        unique = []
        seen = set()
        for model in candidates:
            if model not in seen:
                seen.add(model)
                unique.append(model)

        return sorted(unique, key=self._score_model, reverse=True)[0]

    def _cache_key(self, model: str, task: str, prompt: str) -> str:
        raw = f"{model}\n{task}\n{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _track_success(self, model: str, latency_ms: float) -> None:
        stat = self.memory.model_stats.setdefault(model, ModelStats())
        stat.calls += 1
        stat.latency_ms_total += latency_ms

    def _track_failure(self, model: str) -> None:
        stat = self.memory.model_stats.setdefault(model, ModelStats())
        stat.calls += 1
        stat.failures += 1

    def _call_huggingface(self, model: str, prompt: str, task: str) -> str:
        task_key = task.lower()

        if task_key in {"embedding", "embeddings"}:
            vectors = self.client.feature_extraction(model=model, text=prompt)
            return json.dumps(vectors)

        if task_key == "summarization":
            result = self.client.summarization(model=model, text=prompt)
            return getattr(result, "summary_text", str(result))

        if task_key == "translation":
            result = self.client.translation(model=model, text=prompt)
            return getattr(result, "translation_text", str(result))

        response = self.client.text_generation(
            model=model,
            prompt=prompt,
            max_new_tokens=1024,
        )
        return str(response)

    def generate(self, prompt: str, task: str = "general") -> str:
        self.maybe_refresh_registry()
        self.maybe_improve_router()

        model = self.select_best_model(task=task, prompt=prompt)
        cache_key = self._cache_key(model=model, task=task, prompt=prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]

        attempts = [model]
        attempts.extend(TASK_BACKUPS.get(task.lower(), []))
        attempts.extend(TASK_BACKUPS.get("general", []))

        tried = set()
        for candidate in attempts:
            if candidate in tried:
                continue
            tried.add(candidate)

            for _ in range(MAX_RETRIES + 1):
                started = time.perf_counter()
                try:
                    output = self._call_huggingface(model=candidate, prompt=prompt, task=task)
                    latency_ms = (time.perf_counter() - started) * 1000
                    self._track_success(model=candidate, latency_ms=latency_ms)
                    self.cache[cache_key] = output
                    self._save_memory()
                    return output
                except Exception:
                    self._track_failure(model=candidate)
                    self._save_memory()

        raise RuntimeError("All model attempts failed.")


ROUTER = ModelRouter()


def generate(prompt: str, task: str = "general") -> str:
    """Unified API wrapper for OpenClaw integration points."""
    return ROUTER.generate(prompt=prompt, task=task)


def openclaw_generate(prompt: str, task: str = "general") -> str:
    """Explicit helper so OpenClaw call-sites can replace static model calls."""
    return generate(prompt=prompt, task=task)
