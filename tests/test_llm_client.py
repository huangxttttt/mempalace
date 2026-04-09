from mempalace.llm_client import normalize_base_url


class TestLLMClient:
    def test_normalize_base_url(self):
        assert normalize_base_url("http://localhost:11434/") == "http://localhost:11434"
