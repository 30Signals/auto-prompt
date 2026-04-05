"""
LLM provider setup plus lightweight runtime history / token-cost logging.

This keeps model routing centralized so experiments can switch to a cheaper
provider/model without touching experiment code.
"""

import json
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from .router import get_default_router_slot

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default=None):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_float(name: str, default=None):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _router_config_enabled() -> bool:
    return _env_bool("LLM_USE_ROUTER_CONFIG", False)


def _is_foundry_openai_endpoint(endpoint: str | None) -> bool:
    endpoint = str(endpoint or "").strip().lower()
    return "services.ai.azure.com/openai/v1" in endpoint


def _router_slot_to_config(slot: dict) -> dict:
    provider_map = {
        "google": "gemini",
        "groq": "groq",
        "openrouter": "openrouter",
        "azure": "azure",
        "azure_foundry_openai": "azure_foundry_openai",
        "openai": "openai",
    }
    provider = provider_map.get(slot["type"], slot["type"])
    model = slot["model"]
    if provider == "groq" and "/" not in model:
        model = f"groq/{model}"
    elif provider == "openrouter" and "/" not in model:
        model = f"openrouter/{model}"
    elif provider == "gemini" and "/" not in model:
        model = f"gemini/{model}"
    elif provider == "openai" and "/" not in model:
        model = f"openai/{model}"
    elif provider == "azure" and not model.startswith("azure/"):
        model = f"azure/{model}"
    elif provider == "azure_foundry_openai" and "/" not in model:
        model = f"openai/{model}"
    elif provider == "azure_foundry_openai" and "/" not in model:
        model = f"openai/{model}"

    config = {
        "provider": provider,
        "model": model,
        "api_key": slot["api_key"],
    }
    if slot.get("api_base"):
        config["api_base"] = slot["api_base"]
    return config


def _build_config_from_provider(provider: str) -> dict:
    provider = provider.lower().strip()

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("LLM_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not (api_key and endpoint and deployment):
            raise ValueError(
                "Azure provider selected but AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT, or deployment is missing."
            )
        if _is_foundry_openai_endpoint(endpoint):
            return {
                "provider": "azure_foundry_openai",
                "model": f"openai/{deployment}",
                "api_key": api_key,
                "api_base": endpoint,
            }

        if _is_foundry_openai_endpoint(endpoint):
            return {
                "provider": "azure_foundry_openai",
                "model": f"openai/{deployment}",
                "api_key": api_key,
                "api_base": endpoint,
            }

        return {
            "provider": "azure",
            "model": f"azure/{deployment}",
            "api_key": api_key,
            "api_base": endpoint,
            "api_version": api_version,
        }

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
        if not api_key:
            raise ValueError("Gemini provider selected but GEMINI_API_KEY is missing.")
        if "/" not in model:
            model = f"gemini/{model}"
        return {
            "provider": "gemini",
            "model": model,
            "api_key": api_key,
        }

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")
        if not api_key:
            raise ValueError("Groq provider selected but GROQ_API_KEY is missing.")
        if "/" not in model:
            model = f"groq/{model}"
        return {
            "provider": "groq",
            "model": model,
            "api_key": api_key,
            "api_base": os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1"),
        }

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
        if not api_key:
            raise ValueError("OpenAI provider selected but OPENAI_API_KEY is missing.")
        if "/" not in model:
            model = f"openai/{model}"
        return {
            "provider": "openai",
            "model": model,
            "api_key": api_key,
        }

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("LLM_MODEL")
        if not (api_key and model):
            raise ValueError(
                "OpenRouter provider selected but OPENROUTER_API_KEY or "
                "LLM_MODEL is missing."
            )
        if "/" not in model:
            model = f"openrouter/{model}"
        return {
            "provider": "openrouter",
            "model": model,
            "api_key": api_key,
            "api_base": os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1"),
        }

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_config():
    """
    Resolve provider/model settings from environment variables.

    Priority:
    1. Explicit LLM_PROVIDER
    2. Azure OpenAI
    3. Groq
    4. Gemini
    5. OpenAI
    """
    explicit_provider = os.getenv("LLM_PROVIDER")
    if _router_config_enabled():
        config = _router_slot_to_config(get_default_router_slot())
        config["source"] = "router_config"
    elif explicit_provider:
        config = _build_config_from_provider(explicit_provider)
    elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        config = _build_config_from_provider("azure")
    elif os.getenv("GROQ_API_KEY"):
        config = _build_config_from_provider("groq")
    elif os.getenv("GEMINI_API_KEY"):
        config = _build_config_from_provider("gemini")
    elif os.getenv("OPENAI_API_KEY"):
        config = _build_config_from_provider("openai")
    else:
        raise ValueError(
            "No valid LLM provider config found. Set LLM_PROVIDER explicitly "
            "or provide Azure/Groq/Gemini/OpenAI credentials in .env."
        )

    config["temperature"] = _env_float("LLM_TEMPERATURE", 0.0)
    config["max_tokens"] = _env_int("LLM_MAX_TOKENS", 800)
    config["cache"] = _env_bool("LLM_CACHE", True)
    config["num_retries"] = _env_int("LLM_NUM_RETRIES", 3)
    config["input_cost_per_1m"] = _env_float("LLM_INPUT_COST_PER_1M", 0.0)
    config["output_cost_per_1m"] = _env_float("LLM_OUTPUT_COST_PER_1M", 0.0)
    return config


def setup_dspy_lm(config=None):
    """
    Configure DSPy with the resolved LM config and print a small runtime summary.
    """
    if config is None:
        config = get_llm_config()

    extra_kwargs = {}
    if config.get("api_base"):
        extra_kwargs["api_base"] = config["api_base"]
    if config.get("api_version"):
        extra_kwargs["api_version"] = config["api_version"]

    lm = dspy.LM(
        config["model"],
        api_key=config["api_key"],
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
        cache=config.get("cache", True),
        num_retries=config.get("num_retries", 3),
        **extra_kwargs,
    )

    dspy.settings.configure(lm=lm)
    source = config.get("source", "env")
    print(
        "      LLM:"
        f" source={source}, provider={config['provider']}, model={config['model']},"
        f" cache={config.get('cache', True)}, max_tokens={config.get('max_tokens')}"
    )
    return lm


def _extract_usage_dict(item):
    if not isinstance(item, dict):
        return {}

    direct_usage = item.get("usage")
    if isinstance(direct_usage, dict):
        return direct_usage

    response = item.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return usage

        model_extra = response.get("model_extra")
        if isinstance(model_extra, dict):
            usage = model_extra.get("usage")
            if isinstance(usage, dict):
                return usage

    outputs = item.get("outputs")
    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, dict) and isinstance(output.get("usage"), dict):
                return output["usage"]

    return {}


def _usage_value(usage: dict, *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def save_lm_history_artifacts(output_dir, prefix: str = "dspy_runtime"):
    """
    Save raw LM history plus a compact token/cost summary when usage is available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = dspy.settings.lm
    history = getattr(lm, "history", None)
    if not isinstance(history, list) or not history:
        return None

    raw_path = output_dir / f"{prefix}_history.json"
    raw_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    prompts_txt = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cache_hits = 0

    for idx, item in enumerate(history, start=1):
        prompts_txt.append(f"===== CALL {idx} =====")
        if isinstance(item, dict):
            if item.get("prompt"):
                prompts_txt.append("[prompt]")
                prompts_txt.append(str(item.get("prompt")))
            if item.get("messages"):
                prompts_txt.append("[messages]")
                prompts_txt.append(
                    json.dumps(item.get("messages"), ensure_ascii=False, indent=2, default=str)
                )
            if item.get("response"):
                prompts_txt.append("[response]")
                prompts_txt.append(str(item.get("response")))

            usage = _extract_usage_dict(item)
            p_tokens = _usage_value(
                usage,
                "prompt_tokens",
                "input_tokens",
                "prompt_token_count",
                "input_token_count",
            )
            c_tokens = _usage_value(
                usage,
                "completion_tokens",
                "output_tokens",
                "completion_token_count",
                "output_token_count",
            )
            t_tokens = _usage_value(
                usage,
                "total_tokens",
                "total_token_count",
            ) or (p_tokens + c_tokens)

            prompt_tokens += p_tokens
            completion_tokens += c_tokens
            total_tokens += t_tokens
            if item.get("cache_hit") or usage.get("cache_hit"):
                cache_hits += 1
        else:
            prompts_txt.append(str(item))
        prompts_txt.append("")

    txt_path = output_dir / f"{prefix}_prompts.txt"
    txt_path.write_text("\n".join(prompts_txt), encoding="utf-8")

    config = get_llm_config()
    est_cost = (
        (prompt_tokens / 1000000.0) * config.get("input_cost_per_1m", 0.0)
        + (completion_tokens / 1000000.0) * config.get("output_cost_per_1m", 0.0)
    )
    summary = {
        "provider": config["provider"],
        "model": config["model"],
        "calls": len(history),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_hits": cache_hits,
        "cache_enabled": config.get("cache", True),
        "estimated_cost": est_cost,
        "input_cost_per_1m": config.get("input_cost_per_1m", 0.0),
        "output_cost_per_1m": config.get("output_cost_per_1m", 0.0),
    }

    summary_path = output_dir / f"{prefix}_usage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def enforce_usage_budget(summary: dict | None):
    """Raise if runtime usage exceeds configured token or cost limits."""
    if not summary:
        return

    max_total_tokens = _env_int("LLM_MAX_TOTAL_TOKENS", None)
    max_estimated_cost = _env_float("LLM_MAX_ESTIMATED_COST", None)

    if max_total_tokens is not None and summary.get("total_tokens", 0) > max_total_tokens:
        raise RuntimeError(
            f"LLM usage exceeded token budget: {summary.get('total_tokens', 0)} > {max_total_tokens}"
        )

    if max_estimated_cost is not None and summary.get("estimated_cost", 0.0) > max_estimated_cost:
        raise RuntimeError(
            f"LLM usage exceeded cost budget: {summary.get('estimated_cost', 0.0):.6f} > {max_estimated_cost:.6f}"
        )
