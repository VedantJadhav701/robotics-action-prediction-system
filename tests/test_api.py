"""Unit tests for the FastAPI application"""

import sys
from pathlib import Path

import pytest

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from app.api import app  # noqa: E402


class TestAPIEndpoints:
    """Test suite for API structure and configuration"""

    def test_app_created(self):
        """Test that FastAPI app is created successfully"""
        assert app is not None

    def test_app_has_title(self):
        """Test that API has correct title"""
        assert app.title
        assert "PhysicalAI" in app.title or "Robotics" in app.title

    def test_app_has_version(self):
        """Test that API has version"""
        assert app.version is not None

    def test_app_has_description(self):
        """Test that API has description"""
        assert app.description is not None

    def test_app_routes_exist(self):
        """Test that app has routes defined"""
        assert len(app.routes) > 0

    def test_app_middleware_configured(self):
        """Test that CORS middleware is configured"""
        middleware_types = [type(m).__name__ for m in app.user_middleware]
        # Check if any CORS-related middleware exists
        assert any("CORS" in m or "Cors" in m for m in middleware_types)


class TestAPIIntegration:
    """Integration tests for API structure"""

    def test_api_openapi_schema_generation(self):
        """Test that OpenAPI schema can be generated"""
        openapi_schema = app.openapi()
        assert openapi_schema is not None
        assert "openapi" in openapi_schema or "swagger" in openapi_schema

    def test_api_schema_has_paths(self):
        """Test that OpenAPI schema contains paths"""
        openapi_schema = app.openapi()
        assert "paths" in openapi_schema
        assert len(openapi_schema["paths"]) > 0

    def test_api_schema_has_info(self):
        """Test that OpenAPI schema contains info"""
        openapi_schema = app.openapi()
        assert "info" in openapi_schema
        info = openapi_schema["info"]
        assert "title" in info
        assert "version" in info
