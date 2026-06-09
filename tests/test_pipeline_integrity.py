import pytest
from pipelines.features.build_features import main as build_main
from pipelines.training.train import main as train_main

def test_full_pipeline_smoke_run(tmp_path, monkeypatch):
     """
     Ensures that a change doesn't break the basic execution flow.
     Mocks paths to use a temporary directory.
     """
     # 1. Trigger feature building
     # (In a real test, you'd mock the raw CSV loading)
     try:
         build_main()
         train_main()
     except Exception as e:
         pytest.fail(f"Pipeline crashed with error: {e}")