import importlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def install_dependency_stubs():
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    langchain_openai.AzureChatOpenAI = object
    langchain_openai.AzureOpenAIEmbeddings = object
    langchain_openai.OpenAIEmbeddings = object
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
    openai.OpenAI = object
    sys.modules["openai"] = openai

    sklearn = types.ModuleType("sklearn")
    feature_extraction = types.ModuleType("sklearn.feature_extraction")
    text_module = types.ModuleType("sklearn.feature_extraction.text")

    class FakeHashingVectorizer:
        def __init__(self, *args, **kwargs):
            self.n_features = kwargs.get("n_features", 8)

        def transform(self, texts):
            class Matrix:
                def __init__(self, rows, width):
                    self.rows = rows
                    self.width = width

                def toarray(self):
                    return [[0.0] * self.width for _ in self.rows]

            return Matrix(texts, self.n_features)

    text_module.HashingVectorizer = FakeHashingVectorizer
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.feature_extraction"] = feature_extraction
    sys.modules["sklearn.feature_extraction.text"] = text_module


class WebUIServiceTests(unittest.TestCase):
    def setUp(self):
        install_dependency_stubs()

    def test_config_shape_and_proxy_round_trip(self):
        config_service = importlib.import_module("webui.services.config_service")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = config_service.load_config_state(config_path)

            self.assertIn("llm_configs", config)
            self.assertIn("Local Hashing", config["embedding_configs"])

            config_service.save_proxy_config(config_path, True, "127.0.0.1", "7890")
            self.assertEqual(os.environ["HTTP_PROXY"], "http://127.0.0.1:7890")

            config_service.save_proxy_config(config_path, False, "127.0.0.1", "7890")
            self.assertNotIn("HTTP_PROXY", os.environ)

    def test_chapter_file_sort_load_save(self):
        file_service = importlib.import_module("webui.services.file_service")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_service.save_chapter(tmpdir, 10, "ten")
            file_service.save_chapter(tmpdir, 2, "two")

            self.assertEqual(file_service.list_chapters(tmpdir), ["2", "10"])
            self.assertEqual(file_service.load_chapter(tmpdir, "2"), "two")

    def test_role_library_create_save_move_and_rename_category(self):
        role_service = importlib.import_module("webui.services.role_service")

        with tempfile.TemporaryDirectory() as tmpdir:
            role_service.add_category(tmpdir, "主角团")
            role = role_service.create_role(tmpdir, "主角团", "李青")
            self.assertEqual(role["name"], "李青")

            saved = role_service.save_role(
                tmpdir,
                "主角团",
                "李青",
                "李青",
                {"物品": ["青衫"], "能力": ["剑气"], "状态": ["心理状态: 平静"]},
            )
            self.assertIn("青衫", saved["raw"])

            role_service.add_category(tmpdir, "反派")
            moved = role_service.move_role(tmpdir, "李青", "主角团", "反派")
            self.assertEqual(moved["category"], "反派")

            categories = role_service.rename_category(tmpdir, "反派", "敌对势力")
            self.assertIn("敌对势力", categories)

    def test_webdav_backup_uses_expected_remote_path(self):
        webdav_service = importlib.import_module("webui.services.webdav_service")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            Path(config_path).write_text("{}", encoding="utf-8")

            with mock.patch.object(webdav_service.WebDAVClient, "ensure_directory_exists", return_value=True), \
                    mock.patch.object(webdav_service.WebDAVClient, "upload_file", return_value=True) as upload:
                result = webdav_service.backup_config(config_path, "https://example.com/dav", "u", "p")

            self.assertEqual(result, "配置备份成功。")
            upload.assert_called_once_with(config_path, "AI_Novel_Generator/config.json")

    def test_gradio_app_smoke_if_dependency_installed(self):
        try:
            import gradio  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("gradio is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from webui.app import create_webui; app = create_webui(r'%s'); print(type(app).__name__)"
                    % config_path,
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Blocks", result.stdout)


if __name__ == "__main__":
    unittest.main()
