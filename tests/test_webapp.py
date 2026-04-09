from mempalace.webapp import (
    WebAppState,
    choose_directory_dialog,
    clear_directory_index,
    directory_to_wing,
    format_sources,
    get_directory_index_status,
)


class TestWebAppHelpers:
    def test_format_sources(self):
        citations = [
            {
                "source_file": "notes.txt",
                "wing": "project",
                "room": "planning",
                "similarity": 0.91,
            }
        ]

        rendered = format_sources(citations)

        assert "project / planning / notes.txt" in rendered
        assert "0.91" in rendered

    def test_state_defaults(self):
        state = WebAppState()

        assert state.current_dir == ""
        assert state.current_wing == ""
        assert state.model_base_url == "http://127.0.0.1:11434"
        assert state.model_name == ""
        assert state.last_answer == ""
        assert state.last_hits == []
        assert state.last_context == ""
        assert state.last_sources == []

    def test_directory_to_wing(self):
        assert directory_to_wing(r"E:\Team Docs\My Project") == "my_project"

    def test_choose_directory_dialog_missing_tkinter(self, monkeypatch):
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("missing tkinter")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        try:
            choose_directory_dialog()
        except RuntimeError as exc:
            assert "folder picker" in str(exc).lower()
        else:
            raise AssertionError("choose_directory_dialog should fail without tkinter")

    def test_get_directory_index_status_empty(self):
        result = get_directory_index_status("")
        assert result == {"exists": False, "drawers": 0, "files": 0}

    def test_clear_directory_index_empty(self):
        assert clear_directory_index("") == 0
