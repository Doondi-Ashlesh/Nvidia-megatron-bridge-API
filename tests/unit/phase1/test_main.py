"""Phase 1 unit tests — main.py (CORS branch, CLI entry point)."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestCorsMiddlewareBranch:
    def test_cors_middleware_added_when_origins_set(self, monkeypatch):
        """Lines 89-95 + branch [88→89]: CORS middleware is added when cors_origins_list is truthy.

        We patch the ``settings`` NAME in ``app.main``'s namespace (not the
        Pydantic property on the class) so create_app() sees the mock value.
        """
        import app.main as main_mod
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.cors_origins_list = ["http://localhost:3000"]
        mock_settings.version = "0.1.0-dev"

        monkeypatch.setattr(main_mod, "settings", mock_settings)

        application = main_mod.create_app()
        middleware_class_names = [m.cls.__name__ for m in application.user_middleware]
        assert "CORSMiddleware" in middleware_class_names

    def test_cors_middleware_absent_when_no_origins(self, monkeypatch):
        """Branch [88→96]: CORS middleware is NOT added when cors_origins_list is empty."""
        import app.main as main_mod
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.cors_origins_list = []
        mock_settings.version = "0.1.0-dev"

        monkeypatch.setattr(main_mod, "settings", mock_settings)

        application = main_mod.create_app()
        middleware_class_names = [m.cls.__name__ for m in application.user_middleware]
        assert "CORSMiddleware" not in middleware_class_names


class TestMainCliEntryPoint:
    def test_main_calls_uvicorn_run(self, monkeypatch):
        """Lines 109-111: main() imports uvicorn and calls uvicorn.run with correct args."""
        calls: list[dict] = []

        mock_uvicorn = MagicMock()
        mock_uvicorn.run.side_effect = lambda *a, **kw: calls.append({"args": a, "kwargs": kw})

        # Patch uvicorn in sys.modules so the `import uvicorn` inside main() gets the mock
        monkeypatch.setitem(__import__("sys").modules, "uvicorn", mock_uvicorn)

        from app.main import main
        main()

        assert len(calls) == 1
        assert calls[0]["args"][0] == "app.main:app"
        assert "host" in calls[0]["kwargs"]
        assert "port" in calls[0]["kwargs"]
        assert "log_level" in calls[0]["kwargs"]
