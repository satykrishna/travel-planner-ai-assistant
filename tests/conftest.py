import pytest


@pytest.fixture(autouse=True)
def disable_external_telemetry(monkeypatch):
    # Tests must not send data to a real account even if local credentials exist.
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
