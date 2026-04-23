# config_manager.py
# -*- coding: utf-8 -*-
import copy
import datetime
import json
import os
import threading
import uuid
from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter

LLM_INTERFACE_OPTIONS = ["OpenAI", "Azure OpenAI", "Ollama", "DeepSeek", "Gemini", "ML Studio", "Grok"]
LOCAL_HASHING_EMBEDDING_INTERFACE = "Local Hashing"
EMBEDDING_INTERFACE_OPTIONS = [
    LOCAL_HASHING_EMBEDDING_INTERFACE,
    "OpenAI",
    "Azure OpenAI",
    "Gemini",
    "Ollama",
    "ML Studio",
    "SiliconFlow",
]
GROK_DEFAULT_BASE_URL = "https://api.x.ai/v1"
GROK_DEFAULT_MODEL = "grok-4.20"
OPENAI_DEFAULT_BASE_URLS = {"", "https://api.openai.com", "https://api.openai.com/v1"}
OPENAI_DEFAULT_MODELS = {"", "gpt-4", "gpt-4o-mini", "gpt-5"}


def normalize_embedding_interface(interface_format: str) -> str:
    value = (interface_format or "").strip()
    if value.lower() == "deepseek":
        return LOCAL_HASHING_EMBEDDING_INTERFACE
    return value or "OpenAI"


def get_default_embedding_config(interface_format: str) -> dict:
    interface_format = normalize_embedding_interface(interface_format)
    if interface_format == LOCAL_HASHING_EMBEDDING_INTERFACE:
        return {
            "api_key": "",
            "base_url": "",
            "model_name": "local-hashing",
            "retrieval_k": 4,
            "interface_format": LOCAL_HASHING_EMBEDDING_INTERFACE,
        }
    return {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model_name": "text-embedding-ada-002",
        "retrieval_k": 4,
        "interface_format": interface_format,
    }


def build_default_llm_configs():
    return {
        "默认配置": {
            "id": str(uuid.uuid4()),
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "OpenAI",
            "created_at": datetime.datetime.now().isoformat(),
        },
        "Grok 4.20": {
            "id": str(uuid.uuid4()),
            "api_key": "",
            "base_url": GROK_DEFAULT_BASE_URL,
            "model_name": GROK_DEFAULT_MODEL,
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "Grok",
            "created_at": datetime.datetime.now().isoformat(),
        },
    }


DEFAULT_CONFIG = {
    "last_interface_format": "OpenAI",
    "last_embedding_interface_format": "OpenAI",
    "llm_configs": {
        "DeepSeek V3": {
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "OpenAI",
        },
        "GPT 5": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-5",
            "temperature": 0.7,
            "max_tokens": 32768,
            "timeout": 600,
            "interface_format": "OpenAI",
        },
        "Gemini 2.5 Pro": {
            "api_key": "",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model_name": "gemini-2.5-pro",
            "temperature": 0.7,
            "max_tokens": 32768,
            "timeout": 600,
            "interface_format": "OpenAI",
        },
        "Grok 4.20": {
            "api_key": "",
            "base_url": GROK_DEFAULT_BASE_URL,
            "model_name": GROK_DEFAULT_MODEL,
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "Grok",
        },
    },
    "embedding_configs": {
        "OpenAI": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "text-embedding-ada-002",
            "retrieval_k": 4,
            "interface_format": "OpenAI",
        },
        LOCAL_HASHING_EMBEDDING_INTERFACE: {
            "api_key": "",
            "base_url": "",
            "model_name": "local-hashing",
            "retrieval_k": 4,
            "interface_format": LOCAL_HASHING_EMBEDDING_INTERFACE,
        },
    },
    "other_params": {
        "topic": "",
        "genre": "",
        "num_chapters": 0,
        "word_number": 0,
        "filepath": "",
        "chapter_num": "120",
        "user_guidance": "",
        "characters_involved": "",
        "key_items": "",
        "scene_location": "",
        "time_constraint": "",
    },
    "choose_configs": {
        "prompt_draft_llm": "DeepSeek V3",
        "chapter_outline_llm": "DeepSeek V3",
        "architecture_llm": "Gemini 2.5 Pro",
        "final_chapter_llm": "GPT 5",
        "consistency_review_llm": "DeepSeek V3",
    },
    "proxy_setting": {
        "proxy_url": "127.0.0.1",
        "proxy_port": "",
        "enabled": False,
    },
    "webdav_config": {
        "webdav_url": "",
        "webdav_username": "",
        "webdav_password": "",
    },
}


def get_default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def ensure_config_shape(config: dict) -> dict:
    merged = get_default_config()
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value

    if not merged.get("llm_configs"):
        merged["llm_configs"] = build_default_llm_configs()

    embedding_configs = merged.setdefault("embedding_configs", {})
    for name, emb_conf in list(embedding_configs.items()):
        if isinstance(emb_conf, dict):
            emb_conf["interface_format"] = normalize_embedding_interface(
                emb_conf.get("interface_format", name)
            )

    last_embedding = normalize_embedding_interface(merged.get("last_embedding_interface_format", "OpenAI"))
    merged["last_embedding_interface_format"] = last_embedding
    embedding_configs.setdefault(last_embedding, get_default_embedding_config(last_embedding))

    choose_configs = merged.setdefault("choose_configs", {})
    config_names = list(merged["llm_configs"].keys())
    if config_names:
        for key in [
            "prompt_draft_llm",
            "chapter_outline_llm",
            "architecture_llm",
            "final_chapter_llm",
            "consistency_review_llm",
        ]:
            if choose_configs.get(key) not in merged["llm_configs"]:
                choose_configs[key] = config_names[0]

    merged.setdefault("other_params", {})
    merged.setdefault("proxy_setting", {})
    merged.setdefault("webdav_config", {})
    return merged


def load_config(config_file: str) -> dict:
    """从指定的 config_file 加载配置，若不存在则创建一个默认配置文件。"""

    # PenBo 修改代码，增加配置文件不存在则创建一个默认配置文件
    if not os.path.exists(config_file):
        create_config(config_file)

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return ensure_config_shape(json.load(f))
    except:
            return ensure_config_shape({})


# PenBo 增加了创建默认配置文件函数
def create_config(config_file: str) -> dict:
    """创建一个创建默认配置文件。"""
    config = {
    "last_interface_format": "OpenAI",
    "last_embedding_interface_format": "OpenAI",
    "llm_configs": {
        "DeepSeek V3": {
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "OpenAI"
        },
        "GPT 5": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-5",
            "temperature": 0.7,
            "max_tokens": 32768,
            "timeout": 600,
            "interface_format": "OpenAI"
        },
        "Gemini 2.5 Pro": {
            "api_key": "",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model_name": "gemini-2.5-pro",
            "temperature": 0.7,
            "max_tokens": 32768,
            "timeout": 600,
            "interface_format": "OpenAI"
        },
        "Grok 4.20": {
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
            "model_name": "grok-4.20",
            "temperature": 0.7,
            "max_tokens": 8192,
            "timeout": 600,
            "interface_format": "Grok"
        }
    },
    "embedding_configs": {
        "OpenAI": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "text-embedding-ada-002",
            "retrieval_k": 4,
            "interface_format": "OpenAI"
        },
        "Local Hashing": {
            "api_key": "",
            "base_url": "",
            "model_name": "local-hashing",
            "retrieval_k": 4,
            "interface_format": "Local Hashing"
        }
    },
    "other_params": {
        "topic": "",
        "genre": "",
        "num_chapters": 0,
        "word_number": 0,
        "filepath": "",
        "chapter_num": "120",
        "user_guidance": "",
        "characters_involved": "",
        "key_items": "",
        "scene_location": "",
        "time_constraint": ""
    },
    "choose_configs": {
        "prompt_draft_llm": "DeepSeek V3",
        "chapter_outline_llm": "DeepSeek V3",
        "architecture_llm": "Gemini 2.5 Pro",
        "final_chapter_llm": "GPT 5",
        "consistency_review_llm": "DeepSeek V3"
    },
    "proxy_setting": {
        "proxy_url": "127.0.0.1",
        "proxy_port": "",
        "enabled": False
    },
    "webdav_config": {
        "webdav_url": "",
        "webdav_username": "",
        "webdav_password": ""
    }
}
    save_config(config, config_file)
    return ensure_config_shape(config)



def save_config(config_data: dict, config_file: str) -> bool:
    """将 config_data 保存到 config_file 中，返回 True/False 表示是否成功。"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        return True
    except:
        return False

def test_llm_config(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout, log_func, handle_exception_func):
    """测试当前的LLM配置是否可用"""
    def task():
        try:
            log_func("开始测试LLM配置...")
            llm_adapter = create_llm_adapter(
                interface_format=interface_format,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

            test_prompt = "Please reply 'OK'"
            response = llm_adapter.invoke(test_prompt)
            if response:
                log_func("✅ LLM配置测试成功！")
                log_func(f"测试回复: {response}")
            else:
                log_func("❌ LLM配置测试失败：未获取到响应")
        except Exception as e:
            log_func(f"❌ LLM配置测试出错: {str(e)}")
            handle_exception_func("测试LLM配置时出错")

    threading.Thread(target=task, daemon=True).start()

def test_embedding_config(api_key, base_url, interface_format, model_name, log_func, handle_exception_func):
    """测试当前的Embedding配置是否可用"""
    def task():
        try:
            log_func("开始测试Embedding配置...")
            embedding_adapter = create_embedding_adapter(
                interface_format=interface_format,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )

            test_text = "测试文本"
            embeddings = embedding_adapter.embed_query(test_text)
            if embeddings and len(embeddings) > 0:
                log_func("✅ Embedding配置测试成功！")
                log_func(f"生成的向量维度: {len(embeddings)}")
            else:
                log_func("❌ Embedding配置测试失败：未获取到向量")
        except Exception as e:
            log_func(f"❌ Embedding配置测试出错: {str(e)}")
            handle_exception_func("测试Embedding配置时出错")

    threading.Thread(target=task, daemon=True).start()
