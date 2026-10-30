"""Project paths and configuration. Colab/Drive aware."""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from pathlib import Path
import yaml

DRIVE_ROOT = Path("/content/drive/MyDrive/CreditIQ")


def find_root() -> Path:
    env = os.environ.get("CREDITIQ_ROOT")
    if env:
        return Path(env)
    if DRIVE_ROOT.exists():
        return DRIVE_ROOT
    return Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.raw = self.root / "data/raw"
        self.interim = self.root / "data/interim"
        self.processed = self.root / "data/processed"
        self.models = self.root / "artifacts/models"
        self.metrics = self.root / "artifacts/metrics"
        self.reports = self.root / "artifacts/reports"
        self.figures = self.reports / "figures"
        self.sql = self.root / "sql"
        self.docs = self.root / "docs"
        self.logs = self.root / "logs"
        self.exports = self.root / "exports"
        self.configs = self.root / "configs"
        for p in [self.raw, self.interim, self.processed, self.models, self.metrics,
                  self.reports, self.figures, self.docs, self.logs, self.exports]:
            p.mkdir(parents=True, exist_ok=True)


def get_paths(root: str | Path | None = None) -> Paths:
    return Paths(Path(root) if root else find_root())


def load_config(paths: Paths | None = None) -> dict:
    """Loads configs/config.yaml, or the file named by env CREDITIQ_CONFIG (e.g. configs/config_max.yaml)."""
    paths = paths or get_paths()
    override = os.environ.get("CREDITIQ_CONFIG")
    f = Path(override) if override else paths.configs / "config.yaml"
    if override and not f.is_absolute():
        f = paths.configs / f.name
    with open(f) as fh:
        return yaml.safe_load(fh)


# ---------- phase checkpointing (session recovery) ----------
def _state_file(paths: Paths) -> Path:
    return paths.logs / "pipeline_state.json"


def read_state(paths: Paths) -> dict:
    f = _state_file(paths)
    return json.loads(f.read_text()) if f.exists() else {"completed": {}}


def mark_done(paths: Paths, stage: str, info: dict | None = None) -> None:
    st = read_state(paths)
    st["completed"][stage] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {})}
    st["last_completed"] = stage
    _state_file(paths).write_text(json.dumps(st, indent=2, default=str))


def is_done(paths: Paths, stage: str) -> bool:
    return stage in read_state(paths)["completed"]


def write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())
