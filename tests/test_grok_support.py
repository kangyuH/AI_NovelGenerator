import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def install_llm_dependency_stubs(openai_cls=None):
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    langchain_openai.AzureChatOpenAI = object
    sys.modules["langchain_openai"] = langchain_openai

    google = types.ModuleType("google")
    generativeai = types.ModuleType("google.generativeai")
    generativeai.configure = lambda **kwargs: None
    generativeai.GenerativeModel = object
    generativeai.types = types.SimpleNamespace(GenerationConfig=object)
    sys.modules["google"] = google
    sys.modules["google.generativeai"] = generativeai

    azure = types.ModuleType("azure")
    azure_ai = types.ModuleType("azure.ai")
    azure_inference = types.ModuleType("azure.ai.inference")
    azure_inference.ChatCompletionsClient = object
    azure_core = types.ModuleType("azure.core")
    azure_credentials = types.ModuleType("azure.core.credentials")
    azure_credentials.AzureKeyCredential = object
    azure_models = types.ModuleType("azure.ai.inference.models")
    azure_models.SystemMessage = object
    azure_models.UserMessage = object
    sys.modules["azure"] = azure
    sys.modules["azure.ai"] = azure_ai
    sys.modules["azure.ai.inference"] = azure_inference
    sys.modules["azure.core"] = azure_core
    sys.modules["azure.core.credentials"] = azure_credentials
    sys.modules["azure.ai.inference.models"] = azure_models

    openai = types.ModuleType("openai")
    openai.OpenAI = openai_cls or object
    sys.modules["openai"] = openai


def load_llm_adapters(openai_cls=None):
    sys.modules.pop("llm_adapters", None)
    install_llm_dependency_stubs(openai_cls)
    return importlib.import_module("llm_adapters")


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    instances = []
    response = types.SimpleNamespace(output_text="OK from Grok")

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = FakeResponses(self.response)
        FakeOpenAI.instances.append(self)


class GrokSupportTests(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.instances = []
        FakeOpenAI.response = types.SimpleNamespace(output_text="OK from Grok")

    def test_factory_accepts_grok_case_insensitively(self):
        llm_adapters = load_llm_adapters(FakeOpenAI)

        upper = llm_adapters.create_llm_adapter("Grok", "https://api.x.ai/v1", "grok-4.20", "key", 0.7, 128, 10)
        lower = llm_adapters.create_llm_adapter("grok", "https://api.x.ai/v1", "grok-4.20", "key", 0.7, 128, 10)

        self.assertIsInstance(upper, llm_adapters.GrokAdapter)
        self.assertIsInstance(lower, llm_adapters.GrokAdapter)

    def test_check_base_url_adds_v1_once(self):
        llm_adapters = load_llm_adapters(FakeOpenAI)

        self.assertEqual(llm_adapters.check_base_url("https://api.x.ai"), "https://api.x.ai/v1")
        self.assertEqual(llm_adapters.check_base_url("https://api.x.ai/v1"), "https://api.x.ai/v1")

    def test_grok_invoke_uses_responses_api_and_output_text(self):
        llm_adapters = load_llm_adapters(FakeOpenAI)
        adapter = llm_adapters.GrokAdapter("key", "https://api.x.ai", "", 256, 0.3, 30)

        result = adapter.invoke("Say OK")

        self.assertEqual(result, "OK from Grok")
        client = FakeOpenAI.instances[-1]
        self.assertEqual(client.kwargs["base_url"], "https://api.x.ai/v1")
        self.assertEqual(adapter.model_name, "grok-4.20")
        self.assertEqual(
            client.responses.calls[-1],
            {
                "model": "grok-4.20",
                "input": [{"role": "user", "content": "Say OK"}],
                "max_output_tokens": 256,
                "temperature": 0.3,
                "store": False,
                "timeout": 30,
            },
        )

    def test_grok_invoke_extracts_nested_output_text(self):
        nested_response = types.SimpleNamespace(
            output_text="",
            output=[
                types.SimpleNamespace(type="reasoning", content=[types.SimpleNamespace(text="hidden")]),
                types.SimpleNamespace(
                    type="message",
                    content=[
                        types.SimpleNamespace(type="output_text", text="nested"),
                        {"type": "output_text", "text": "text"},
                    ],
                ),
            ],
        )
        FakeOpenAI.response = nested_response
        llm_adapters = load_llm_adapters(FakeOpenAI)
        adapter = llm_adapters.GrokAdapter("key", "https://api.x.ai/v1", "grok-custom", 128, 0.7, 10)

        self.assertEqual(adapter.invoke("prompt"), "nested\ntext")

    def test_created_default_config_includes_grok_without_changing_choices(self):
        fake_llm_adapters = types.ModuleType("llm_adapters")
        fake_llm_adapters.create_llm_adapter = lambda *args, **kwargs: None
        fake_embedding_adapters = types.ModuleType("embedding_adapters")
        fake_embedding_adapters.create_embedding_adapter = lambda *args, **kwargs: None
        sys.modules["llm_adapters"] = fake_llm_adapters
        sys.modules["embedding_adapters"] = fake_embedding_adapters
        sys.modules.pop("config_manager", None)
        config_manager = importlib.import_module("config_manager")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_manager.create_config(str(config_path))
            data = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(data["llm_configs"]["Grok 4.20"]["base_url"], "https://api.x.ai/v1")
        self.assertEqual(data["llm_configs"]["Grok 4.20"]["model_name"], "grok-4.20")
        self.assertEqual(data["llm_configs"]["Grok 4.20"]["interface_format"], "Grok")
        self.assertEqual(data["choose_configs"]["prompt_draft_llm"], "DeepSeek V3")
        self.assertEqual(data["choose_configs"]["final_chapter_llm"], "GPT 5")


if __name__ == "__main__":
    unittest.main()
