import importlib
import sys
import types
import unittest


def load_embedding_adapters():
    fake_langchain_openai = types.ModuleType("langchain_openai")
    fake_langchain_openai.AzureOpenAIEmbeddings = object
    fake_langchain_openai.OpenAIEmbeddings = object
    sys.modules["langchain_openai"] = fake_langchain_openai
    sys.modules.pop("embedding_adapters", None)
    return importlib.import_module("embedding_adapters")


class LocalHashingEmbeddingTests(unittest.TestCase):
    def test_factory_accepts_local_hashing(self):
        embedding_adapters = load_embedding_adapters()

        adapter = embedding_adapters.create_embedding_adapter(
            "Local Hashing",
            "",
            "",
            "local-hashing"
        )

        self.assertIsInstance(adapter, embedding_adapters.LocalHashingEmbeddingAdapter)

    def test_factory_accepts_local_hashing_aliases(self):
        embedding_adapters = load_embedding_adapters()

        for interface_name in ["Local", "Hashing"]:
            with self.subTest(interface_name=interface_name):
                adapter = embedding_adapters.create_embedding_adapter(
                    interface_name,
                    "",
                    "",
                    "local-hashing"
                )
                self.assertIsInstance(adapter, embedding_adapters.LocalHashingEmbeddingAdapter)

    def test_embed_query_returns_fixed_dimension_vector(self):
        embedding_adapters = load_embedding_adapters()
        adapter = embedding_adapters.create_embedding_adapter("Local Hashing", "", "", "local-hashing")

        vector = adapter.embed_query("测试文本")

        self.assertEqual(len(vector), embedding_adapters.LOCAL_HASHING_EMBEDDING_DIM)

    def test_embed_documents_returns_fixed_dimension_vectors(self):
        embedding_adapters = load_embedding_adapters()
        adapter = embedding_adapters.create_embedding_adapter("Local Hashing", "", "", "local-hashing")

        vectors = adapter.embed_documents(["甲", "乙"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), embedding_adapters.LOCAL_HASHING_EMBEDDING_DIM)
        self.assertEqual(len(vectors[1]), embedding_adapters.LOCAL_HASHING_EMBEDDING_DIM)

    def test_deepseek_is_not_treated_as_embedding_backend(self):
        embedding_adapters = load_embedding_adapters()

        with self.assertRaises(ValueError):
            embedding_adapters.create_embedding_adapter(
                "DeepSeek",
                "key",
                "https://api.deepseek.com/v1",
                "deepseek-chat"
            )


if __name__ == "__main__":
    unittest.main()
