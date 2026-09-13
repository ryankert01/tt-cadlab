"""The kernel-guide figures regenerate deterministically and are well-formed SVG (CPU only)."""
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import kernel_figures


def test_every_figure_renders_to_well_formed_svg(tmp_path):
    written = kernel_figures.write_all(tmp_path)
    assert {p.stem for p in written} == set(kernel_figures.FIGURES)
    for p in written:
        root = ET.fromstring(p.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
        assert root.get("viewBox"), p.name
        # every figure carries a caption line
        assert any("Fig." in (t.text or "") for t in root.iter() if t.tag.endswith("text")), p.name


def test_committed_figures_match_the_generator(tmp_path):
    """docs/figures/*.svg is what scripts/kernel_figures.py produces today (re-run it if this fails)."""
    written = kernel_figures.write_all(tmp_path)
    committed = Path(kernel_figures.OUT)
    for p in written:
        assert (committed / p.name).read_text(encoding="utf-8") == p.read_text(encoding="utf-8"), p.name
