"""Standalone Ollama text-generation client for behavior agent JSON analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class OllamaGenerateResult:
    ok: bool
    model: str
    data: Optional[Dict[str, Any]] = None
    error: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0


class OllamaBehaviorClient:
    """Robust, retrying client around Ollama `/api/generate`."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout_seconds: float,
        output_dir: str | Path,
        temperature: float = 0.1,
        num_predict: int = 768,
        num_ctx: int = 4096,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.num_ctx = int(num_ctx)
        self.output_dir = Path(output_dir)
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, prompt: str) -> OllamaGenerateResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }

        start = time.perf_counter()
        last_error = ""
        for attempt in range(2):
            try:
                response = self._post_json("/api/generate", payload)
            except Exception as exc:
                last_error = str(exc)
                if attempt == 0:
                    continue
                return OllamaGenerateResult(
                    ok=False,
                    model=self.model,
                    error=f"ollama_request_failed: {last_error}",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )

            raw_text = str(response.get("response", "")).strip()
            try:
                parsed = _parse_json_object(raw_text)
            except Exception as exc:
                self._write_invalid_debug(raw_text=raw_text, response=response, error=str(exc))
                return OllamaGenerateResult(
                    ok=False,
                    model=self.model,
                    error=f"invalid_json_response: {exc}",
                    raw_response=raw_text,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            if not isinstance(parsed, dict):
                self._write_invalid_debug(raw_text=raw_text, response=response, error="parsed_json_not_object")
                return OllamaGenerateResult(
                    ok=False,
                    model=self.model,
                    error="invalid_json_response: parsed_json_not_object",
                    raw_response=raw_text,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )

            return OllamaGenerateResult(
                ok=True,
                model=self.model,
                data=parsed,
                raw_response=raw_text,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        return OllamaGenerateResult(
            ok=False,
            model=self.model,
            error=f"ollama_request_failed: {last_error}",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=f"{self.host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"ollama_unreachable: {exc}") from exc
        return json.loads(text)

    def _write_invalid_debug(self, *, raw_text: str, response: Dict[str, Any], error: str) -> None:
        ts = int(time.time() * 1000)
        path = self.debug_dir / f"invalid_response_{ts}.txt"
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(f"model={self.model}\n")
                f.write(f"error={error}\n")
                f.write("raw_response_text:\n")
                f.write(raw_text)
                f.write("\n\nfull_ollama_response:\n")
                f.write(json.dumps(response, indent=2, ensure_ascii=False))
        except OSError:
            return


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
