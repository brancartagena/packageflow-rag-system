"""Lets scripts import packageflow_rag without installing it (adds src/ to sys.path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
