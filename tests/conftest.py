import os, shutil, sys
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Synthetic-data project run through every stage (smoke test of the code, not of model quality)."""
    root = tmp_path_factory.mktemp("ciq")
    for d in ("configs", "sql", "dashboard"): shutil.copytree(REPO / d, root / d)
    cfg = yaml.safe_load((root / "configs/config.yaml").read_text())
    cfg["models"] = ["dummy", "logreg", "random_forest", "xgboost", "lightgbm"]; cfg["cv"] = {"folds": 3, "sample_n": None}
    (root / "configs/config.yaml").write_text(yaml.safe_dump(cfg))
    os.environ["CREDITIQ_ROOT"] = str(root)
    from creditiq import pipeline
    paths = pipeline.run("all", root=str(root), synthetic=True, n_train=3000)
    return paths


@pytest.fixture(scope="session")
def cfg(project):
    from creditiq.config import load_config
    return load_config(project)
