"""Smoke test: src.data is importable from the project root and exposes load_all."""
import src.data as data


def test_import_and_api():
    assert callable(data.load_all)
    assert data.parse_date.__name__ == "parse_date"
