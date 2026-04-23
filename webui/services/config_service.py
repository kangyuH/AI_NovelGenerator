# -*- coding: utf-8 -*-
import datetime
import os
import uuid

from config_manager import (
    EMBEDDING_INTERFACE_OPTIONS,
    LLM_INTERFACE_OPTIONS,
    build_default_llm_configs,
    get_default_embedding_config,
    load_config,
    normalize_embedding_interface,
    save_config,
)
from embedding_adapters import create_embedding_adapter
from llm_adapters import create_llm_adapter

from .file_service import safe_float, safe_int

ROLE_KEYS = [
    "architecture_llm",
    "chapter_outline_llm",
    "prompt_draft_llm",
    "final_chapter_llm",
    "consistency_review_llm",
]


def load_config_state(config_file: str) -> dict:
    return load_config(config_file)


def config_names(config: dict) -> list[str]:
    names = list(config.get("llm_configs", {}).keys())
    if not names:
        config["llm_configs"] = build_default_llm_configs()
        names = list(config["llm_configs"].keys())
    return names


def embedding_names(config: dict) -> list[str]:
    configured = list(config.get("embedding_configs", {}).keys())
    names = []
    for name in [*EMBEDDING_INTERFACE_OPTIONS, *configured]:
        normalized = normalize_embedding_interface(name)
        if normalized not in names:
            names.append(normalized)
    return names


def selected_embedding_name(config: dict) -> str:
    return normalize_embedding_interface(config.get("last_embedding_interface_format", "OpenAI"))


def get_llm_config(config: dict, name: str) -> dict:
    llm_configs = config.get("llm_configs", {})
    if name not in llm_configs:
        names = config_names(config)
        if not names:
            raise ValueError("未找到可用的大模型配置。")
        name = names[0]
    return llm_configs[name]


def get_llm_config_for_role(config: dict, role_key: str) -> dict:
    choose_configs = config.get("choose_configs", {})
    return get_llm_config(config, choose_configs.get(role_key, ""))


def get_embedding_config(config: dict, name: str | None = None) -> dict:
    name = normalize_embedding_interface(name or selected_embedding_name(config))
    configs = config.setdefault("embedding_configs", {})
    if name not in configs:
        configs[name] = get_default_embedding_config(name)
    return configs[name]


def save_project_params(config_file: str, params: dict) -> dict:
    config = load_config(config_file)
    config["other_params"] = {
        "topic": params.get("topic", ""),
        "genre": params.get("genre", ""),
        "num_chapters": safe_int(params.get("num_chapters"), 10),
        "word_number": safe_int(params.get("word_number"), 3000),
        "filepath": params.get("filepath", ""),
        "chapter_num": str(safe_int(params.get("chapter_num"), 1)),
        "user_guidance": params.get("user_guidance", ""),
        "characters_involved": params.get("characters_involved", ""),
        "key_items": params.get("key_items", ""),
        "scene_location": params.get("scene_location", ""),
        "time_constraint": params.get("time_constraint", ""),
    }
    save_config(config, config_file)
    return config


def save_llm_config(config_file: str, name: str, values: dict) -> dict:
    config = load_config(config_file)
    name = (name or "").strip()
    if not name:
        raise ValueError("配置名称不能为空。")
    config.setdefault("llm_configs", {})
    config["llm_configs"].setdefault(name, {"id": str(uuid.uuid4()), "created_at": datetime.datetime.now().isoformat()})
    config["llm_configs"][name].update({
        "api_key": values.get("api_key", ""),
        "base_url": values.get("base_url", ""),
        "model_name": values.get("model_name", ""),
        "temperature": safe_float(values.get("temperature"), 0.7),
        "max_tokens": safe_int(values.get("max_tokens"), 8192),
        "timeout": safe_int(values.get("timeout"), 600),
        "interface_format": values.get("interface_format", "OpenAI"),
        "updated_at": datetime.datetime.now().isoformat(),
    })
    config["last_interface_format"] = config["llm_configs"][name]["interface_format"]
    save_config(config, config_file)
    return config


def add_llm_config(config_file: str, name: str) -> dict:
    config = load_config(config_file)
    name = (name or "").strip()
    if not name:
        raise ValueError("新配置名称不能为空。")
    if name in config.get("llm_configs", {}):
        raise ValueError(f"配置 '{name}' 已存在。")
    config.setdefault("llm_configs", {})[name] = {
        "id": str(uuid.uuid4()),
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 8192,
        "timeout": 600,
        "interface_format": "OpenAI",
        "created_at": datetime.datetime.now().isoformat(),
    }
    save_config(config, config_file)
    return config


def rename_llm_config(config_file: str, old_name: str, new_name: str) -> dict:
    config = load_config(config_file)
    old_name = (old_name or "").strip()
    new_name = (new_name or "").strip()
    if old_name not in config.get("llm_configs", {}):
        raise ValueError("当前配置不存在。")
    if not new_name:
        raise ValueError("新配置名称不能为空。")
    if new_name != old_name and new_name in config["llm_configs"]:
        raise ValueError(f"配置 '{new_name}' 已存在。")
    config["llm_configs"][new_name] = config["llm_configs"].pop(old_name)
    for key, value in config.get("choose_configs", {}).items():
        if value == old_name:
            config["choose_configs"][key] = new_name
    save_config(config, config_file)
    return config


def delete_llm_config(config_file: str, name: str) -> dict:
    config = load_config(config_file)
    name = (name or "").strip()
    llm_configs = config.get("llm_configs", {})
    if name not in llm_configs:
        raise ValueError("当前配置不存在。")
    if len(llm_configs) <= 1:
        raise ValueError("至少需要保留一个大模型配置。")
    del llm_configs[name]
    fallback = next(iter(llm_configs))
    for key, value in config.get("choose_configs", {}).items():
        if value == name:
            config["choose_configs"][key] = fallback
    save_config(config, config_file)
    return config


def save_embedding_config(config_file: str, values: dict) -> dict:
    config = load_config(config_file)
    interface = normalize_embedding_interface(values.get("interface_format", "OpenAI"))
    config.setdefault("embedding_configs", {})[interface] = {
        "api_key": values.get("api_key", ""),
        "base_url": values.get("base_url", ""),
        "model_name": values.get("model_name", ""),
        "retrieval_k": safe_int(values.get("retrieval_k"), 4),
        "interface_format": interface,
    }
    config["last_embedding_interface_format"] = interface
    save_config(config, config_file)
    return config


def save_choose_configs(config_file: str, values: dict) -> dict:
    config = load_config(config_file)
    names = set(config_names(config))
    config.setdefault("choose_configs", {})
    for key in ROLE_KEYS:
        value = values.get(key)
        if value not in names:
            raise ValueError(f"模型用途配置无效: {key}")
        config["choose_configs"][key] = value
    save_config(config, config_file)
    return config


def apply_proxy_setting(proxy_setting: dict):
    if proxy_setting.get("enabled"):
        address = proxy_setting.get("proxy_url", "127.0.0.1")
        port = proxy_setting.get("proxy_port", "")
        os.environ["HTTP_PROXY"] = f"http://{address}:{port}"
        os.environ["HTTPS_PROXY"] = f"http://{address}:{port}"
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)


def save_proxy_config(config_file: str, enabled: bool, address: str, port: str) -> dict:
    config = load_config(config_file)
    config["proxy_setting"] = {
        "enabled": bool(enabled),
        "proxy_url": (address or "").strip() or "127.0.0.1",
        "proxy_port": (port or "").strip(),
    }
    apply_proxy_setting(config["proxy_setting"])
    save_config(config, config_file)
    return config


def test_llm_config_sync(values: dict) -> str:
    adapter = create_llm_adapter(
        interface_format=values.get("interface_format", "OpenAI"),
        base_url=values.get("base_url", ""),
        model_name=values.get("model_name", ""),
        api_key=values.get("api_key", ""),
        temperature=safe_float(values.get("temperature"), 0.7),
        max_tokens=safe_int(values.get("max_tokens"), 8192),
        timeout=safe_int(values.get("timeout"), 600),
    )
    response = adapter.invoke("Please reply 'OK'")
    return f"LLM 配置测试成功。测试回复: {response}" if response else "LLM 配置测试失败：未获取到响应。"


def test_embedding_config_sync(values: dict) -> str:
    adapter = create_embedding_adapter(
        interface_format=normalize_embedding_interface(values.get("interface_format", "OpenAI")),
        api_key=values.get("api_key", ""),
        base_url=values.get("base_url", ""),
        model_name=values.get("model_name", ""),
    )
    vector = adapter.embed_query("测试文本")
    return f"Embedding 配置测试成功。向量维度: {len(vector)}" if vector else "Embedding 配置测试失败：未获取到向量。"

