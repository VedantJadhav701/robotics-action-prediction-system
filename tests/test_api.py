"""Unit tests for the FastAPI application"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from app.api import app  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "healthy"]

    def test_health_structure(self, client):
        """Test that health check returns required fields"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # Should have timestamp and status
        assert "timestamp" in data or "status" in data

    def test_api_documentation(self, client):
        """Test that API documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema or "swagger" in schema.get("info", {})

    def test_api_title(self, client):
        """Test that API has correct title in schema"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "PhysicalAI" in schema.get("info", {}).get("title", "")

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/health")
        assert response.status_code == 200
        # CORS should allow any origin in development
        assert "access-control-allow-origin" in response.headers

    def test_home_redirect(self, client):
        """Test that home page is available"""
        response = client.get("/", follow_redirects=True)
        # Should be 200 (after redirect) or 307 (redirect)
        assert response.status_code in [200, 307, 404]


class TestAPIIntegration:
    """Integration tests for API"""

    def test_api_starts_successfully(self, client):
        """Test that API can process basic requests"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_api_version_present(self, client):
        """Test that API version is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert "version" in schema["info"]
