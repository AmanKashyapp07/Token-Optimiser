import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_health(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
