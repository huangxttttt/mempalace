from mempalace.answerer import (
    ask_memories,
    build_chat_messages,
    build_context_block,
    build_qa_messages,
    synthesize_answer,
)


class TestAnswerer:
    def test_ask_memories_returns_answer_and_citations(self, palace_path, seeded_collection):
        result = ask_memories("What handles JWT authentication?", palace_path, n_results=3)

        assert result["question"] == "What handles JWT authentication?"
        assert result["answer"] is not None
        assert "JWT" in result["answer"] or "authentication" in result["answer"].lower()
        assert len(result["citations"]) > 0
        assert result["citations"][0]["source_file"] == "auth.py"

    def test_ask_memories_no_hits(self, palace_path, seeded_collection):
        result = ask_memories("quantum entanglement in deep space", palace_path, n_results=2)

        assert result["answer"] is None
        assert result["citations"] == []
        assert result["results"] == []

    def test_synthesize_answer_falls_back_without_keyword_overlap(self):
        hits = [
            {
                "text": "First line with useful context.\nSecond line with more details.",
                "wing": "project",
                "room": "general",
                "source_file": "notes.txt",
                "similarity": 0.8,
            }
        ]

        answer, citations = synthesize_answer("zzz unmatched tokens", hits)

        assert "First line with useful context." in answer
        assert citations == hits

    def test_build_context_block(self):
        hits = [
            {
                "text": "JWT is used for authentication.",
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "similarity": 0.91,
            }
        ]

        context = build_context_block(hits)

        assert "[Source 1]" in context
        assert "file: auth.py" in context
        assert "JWT is used for authentication." in context

    def test_build_qa_messages(self):
        hits = [
            {
                "text": "JWT is used for authentication.",
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "similarity": 0.91,
            }
        ]

        messages = build_qa_messages("为什么用 JWT？", hits)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "为什么用 JWT？" in messages[1]["content"]
        assert "[Source 1]" in messages[1]["content"]

    def test_build_chat_messages_with_history(self):
        hits = [
            {
                "text": "JWT is used for authentication.",
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "similarity": 0.91,
            }
        ]
        history = [
            {"question": "登录方案是什么？", "answer": "当前使用 JWT。"},
            {"question": "为什么这样做？", "answer": "因为移动端调用更直接。"},
        ]

        messages = build_chat_messages(question="刷新 token 怎么处理？", results=hits, history=history)

        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "登录方案是什么？"
        assert messages[2]["content"] == "当前使用 JWT。"
        assert messages[-1]["role"] == "user"
        assert "刷新 token 怎么处理？" in messages[-1]["content"]
