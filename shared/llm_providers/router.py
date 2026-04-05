"""
LiteLLM router utilities for cheap-model routing and provider failover.

This module is intentionally independent from DSPy so it can be reused for:
- raw chat/tool calls
- future non-DSPy pipelines
- reading a shared router config for DSPy model selection
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def find_router_config_path(config_path: str | Path | None = None) -> Path | None:
    candidates = []
    if config_path:
        candidates.append(Path(config_path))

    env_path = os.getenv("LLM_ROUTER_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path.home() / ".llm_router_config.json")
    candidates.append(Path.cwd() / "llm_router_config.json")

    for path in candidates:
        if path.exists():
            return path
    return None


def load_router_config(config_path: str | Path | None = None) -> dict:
    path = find_router_config_path(config_path)
    if path is None:
        raise FileNotFoundError(
            "No llm_router_config.json found. Checked env/home/current directory."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def expand_router_slots(providers: list[dict]) -> list[dict]:
    """
    Expand provider/model pairs into a flat slot list.
    """
    slots = []
    for provider in providers:
        if not provider.get("enabled", True):
            continue

        provider_name = provider["name"]
        provider_type = provider["type"]
        provider_api_key = provider["api_key"]

        for entry in provider.get("models", []):
            if isinstance(entry, dict):
                if not entry.get("enabled", True):
                    continue
                model = entry["model"]
                slot_extra = {k: v for k, v in entry.items() if k not in {"model", "enabled"}}
            else:
                model = entry
                slot_extra = {}

            slots.append(
                {
                    "name": f"{provider_name}::{model}",
                    "provider_name": provider_name,
                    "type": provider_type,
                    "api_key": provider_api_key,
                    "model": model,
                    **slot_extra,
                }
            )
    return slots


def get_default_router_slot(config_path: str | Path | None = None) -> dict:
    """
    Return the first enabled slot from router config.
    """
    config = load_router_config(config_path)
    slots = expand_router_slots(config.get("providers", []))
    if not slots:
        raise ValueError("No enabled provider/model slots found in router config.")
    return slots[0]


class LLMRouter:
    """
    LiteLLM-based router with cooldowns, retries, and slot failover.
    """

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = find_router_config_path(config_path)
        self._config = load_router_config(self._config_path)
        self._slots = expand_router_slots(self._config["providers"])
        self._slot_index = 0
        self._cooldown_until: dict[str, datetime] = {}
        self._disabled: set[str] = set()
        self._failure_count: dict[str, int] = {}

        if not self._slots:
            raise ValueError("No provider slots found in router config.")

        logger.info(
            "LLMRouter initialized with %s slots from %s",
            len(self._slots),
            self._config_path,
        )

    def _slot_key(self, slot: dict) -> str:
        return slot["name"]

    def _is_available(self, slot: dict) -> bool:
        key = self._slot_key(slot)
        if key in self._disabled:
            return False
        cooldown = self._cooldown_until.get(key)
        if cooldown and datetime.now() < cooldown:
            return False
        return True

    def _next_available_slot(self) -> dict | None:
        n = len(self._slots)
        for _ in range(n):
            slot = self._slots[self._slot_index % n]
            self._slot_index = (self._slot_index + 1) % n
            if self._is_available(slot):
                return slot
        return None

    def _wait_for_slot(self) -> dict:
        while True:
            slot = self._next_available_slot()
            if slot:
                return slot

            active_cooldowns = [
                cd
                for key, cd in self._cooldown_until.items()
                if key not in self._disabled and datetime.now() < cd
            ]
            if not active_cooldowns:
                raise RuntimeError(
                    "All LLM slots are permanently disabled. Check router config/API keys."
                )

            soonest = min(active_cooldowns)
            wait_secs = (soonest - datetime.now()).total_seconds() + 0.5
            print(f"  [router] All slots cooling down. Waiting {wait_secs:.1f}s...")
            time.sleep(max(0.5, wait_secs))

    def _set_cooldown(self, slot: dict, base_seconds: int, max_seconds: int = 600):
        key = self._slot_key(slot)
        failures = self._failure_count.get(key, 0)
        seconds = min(base_seconds * (2 ** failures), max_seconds)
        self._failure_count[key] = failures + 1
        self._cooldown_until[key] = datetime.now() + timedelta(seconds=seconds)
        logger.warning("Slot %s on cooldown for %ss", key, seconds)
        print(f"    cooldown {seconds}s (attempt #{failures + 1})")

    def _disable_slot(self, slot: dict, reason: str):
        key = self._slot_key(slot)
        self._disabled.add(key)
        logger.error("Slot %s permanently disabled: %s", key, reason)
        print(f"  [router] Slot {slot['name']} disabled: {reason}")

    def _build_litellm_kwargs(self, slot: dict, **extra) -> tuple[str, dict]:
        kwargs = dict(extra)
        provider_type = slot["type"]

        if provider_type == "openrouter":
            model_str = f"openrouter/{slot['model']}"
            kwargs["api_base"] = slot.get("api_base", "https://openrouter.ai/api/v1")
            kwargs["api_key"] = slot["api_key"]
        elif provider_type == "google":
            model_str = slot["model"]
            kwargs["api_key"] = slot["api_key"]
        elif provider_type == "groq":
            model_str = slot["model"]
            kwargs["api_key"] = slot["api_key"]
            if slot.get("api_base"):
                kwargs["api_base"] = slot["api_base"]
        else:
            model_str = slot["model"]
            kwargs["api_key"] = slot["api_key"]
            if slot.get("api_base"):
                kwargs["api_base"] = slot["api_base"]

        return model_str, kwargs

    def _parse_xml_tool_calls(self, error_str: str, tools: list[dict]) -> object | None:
        failed_gen = ""
        json_match = re.search(r"\{.*\}", error_str, re.DOTALL)
        if json_match:
            try:
                err_data = json.loads(json_match.group())
                failed_gen = err_data.get("error", {}).get("failed_generation", "")
            except (json.JSONDecodeError, AttributeError):
                failed_gen = ""

        if not failed_gen:
            return None

        calls = re.findall(r"<function=(\w+)>(.*?)</function>", failed_gen, re.DOTALL)
        if not calls:
            return None

        first_param = {}
        for tool in tools:
            fn = tool["function"]
            required = fn["parameters"].get("required", [])
            first_param[fn["name"]] = required[0] if required else None

        class _Fn:
            def __init__(self, name: str, arguments: str):
                self.name = name
                self.arguments = arguments

        class _TC:
            def __init__(self, name: str, arguments: str):
                self.id = f"call_{uuid.uuid4().hex[:8]}"
                self.type = "function"
                self.function = _Fn(name, arguments)

        class _Msg:
            def __init__(self, tool_calls):
                self.content = ""
                self.tool_calls = tool_calls

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Response:
            def __init__(self, choices):
                self.choices = choices

        tool_calls = []
        for name, content in calls:
            content = content.strip()
            try:
                args = json.loads(content)
            except json.JSONDecodeError:
                param = first_param.get(name)
                args = {param: content} if param else {}
            tool_calls.append(_TC(name, json.dumps(args)))

        print(
            "  [router] Parsed XML tool calls:"
            f" {[tc.function.name for tc in tool_calls]}"
        )
        return _Response([_Choice(_Msg(tool_calls))])

    def _call_with_retry(self, build_kwargs_fn, tools=None) -> object:
        while True:
            slot = self._wait_for_slot()
            model_str, kwargs = build_kwargs_fn(slot)

            try:
                response = litellm.completion(model=model_str, **kwargs)
                self._failure_count.pop(self._slot_key(slot), None)
                return response

            except RateLimitError:
                print(f"  [router] Rate limit on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=60, max_seconds=600)

            except ServiceUnavailableError:
                print(f"  [router] Service unavailable on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=30, max_seconds=300)

            except APIConnectionError:
                print(f"  [router] Connection error on {slot['name']}", end="")
                self._set_cooldown(slot, base_seconds=10, max_seconds=120)

            except AuthenticationError as e:
                self._disable_slot(slot, f"AuthenticationError: {e}")

            except NotFoundError as e:
                err_str = str(e).lower()
                if "guardrail" in err_str or "data policy" in err_str or "privacy" in err_str:
                    self._disable_slot(slot, f"Account restriction: {e}")
                elif "no endpoints" in err_str:
                    self._disable_slot(slot, f"No endpoints available: {e}")
                else:
                    print(f"  [router] Not found on {slot['name']}", end="")
                    self._set_cooldown(slot, base_seconds=60, max_seconds=600)

            except ContextWindowExceededError:
                raise

            except BadRequestError as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in ("decommissioned", "not supported", "deprecated")):
                    self._disable_slot(slot, f"Model unavailable: {e}")
                elif "tool_use_failed" in err_str or "failed_generation" in err_str:
                    if tools:
                        synthetic = self._parse_xml_tool_calls(str(e), tools)
                        if synthetic:
                            return synthetic
                    print(f"  [router] Tool generation failed on {slot['name']}", end="")
                    self._set_cooldown(slot, base_seconds=15, max_seconds=120)
                else:
                    raise

            except Exception as e:
                print(
                    f"  [router] Unexpected error on {slot['name']}:"
                    f" {type(e).__name__}: {e}",
                    end="",
                )
                self._set_cooldown(slot, base_seconds=10, max_seconds=120)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens: int = 4096,
    ) -> object:
        def build_kwargs(slot):
            model_str, kwargs = self._build_litellm_kwargs(slot)
            kwargs["messages"] = messages
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["max_tokens"] = max_tokens
            return model_str, kwargs

        return self._call_with_retry(build_kwargs, tools=tools)

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 4096) -> object:
        def build_kwargs(slot):
            model_str, kwargs = self._build_litellm_kwargs(slot)
            kwargs["messages"] = messages
            kwargs["max_tokens"] = max_tokens
            return model_str, kwargs

        return self._call_with_retry(build_kwargs)
