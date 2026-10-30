"""Config override: CREDITIQ_CONFIG selects an alternative file in configs/."""
from creditiq.config import get_paths, load_config


def test_default_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDITIQ_CONFIG", raising=False)
    cfg = load_config(get_paths())
    assert cfg["tuning"]["optuna_trials"] == 0 and cfg["cv"]["sample_n"] == 120000


def test_max_config_override(monkeypatch):
    monkeypatch.setenv("CREDITIQ_CONFIG", "config_max.yaml")
    cfg = load_config(get_paths())
    assert cfg["tuning"]["optuna_trials"] > 0 and cfg["cv"]["sample_n"] is None
    assert "random_forest" not in cfg["models"]
