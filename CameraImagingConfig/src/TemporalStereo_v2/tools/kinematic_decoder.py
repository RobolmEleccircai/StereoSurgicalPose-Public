"""Compatibility imports; canonical inference code is in pose_display."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_display.decoding import OUTPUT_LAYOUT, decode
from pose_display.projection import project
