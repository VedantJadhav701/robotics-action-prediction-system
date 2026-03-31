"""Test that all project modules can be imported successfully"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports:
    """Test suite for module imports"""

    def test_model_import(self):
        """Test that model module can be imported"""
        from src.model import RoboticsLSTM

        assert RoboticsLSTM is not None

    def test_data_loader_import(self):
        """Test that data_loader module can be imported"""
        from src.data_loader import load_parquet_files

        assert load_parquet_files is not None

    def test_monitoring_import(self):
        """Test that monitoring module can be imported"""
        from src.monitoring import MetricsCollector, PerformanceMonitor

        assert MetricsCollector is not None
        assert PerformanceMonitor is not None

    def test_api_import(self):
        """Test that API module can be imported"""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.api import app

        assert app is not None

    def test_dependencies_available(self):
        """Test that required dependencies are installed"""
        import numpy
        import pandas
        import torch
        from fastapi import FastAPI
        from pydantic import BaseModel

        assert numpy is not None
        assert pandas is not None
        assert torch is not None
        assert FastAPI is not None
        assert BaseModel is not None
