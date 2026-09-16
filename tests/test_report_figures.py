"""The report figures (docs/report_palu_on_tt.md) regenerate deterministically and are well-formed SVG (CPU only)."""
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import report_figures


def test_every_report_figure_renders_to_well_formed_svg(tmp_path):
    written = report_figures.write_all(tmp_path)
    assert {p.stem for p in written} == set(report_figures.FIGURES)
    for p in written:
        root = ET.fromstring(p.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
        assert root.get("viewBox"), p.name
        assert any("Fig. R" in (t.text or "") for t in root.iter() if t.tag.endswith("text")), p.name


def test_committed_report_figures_match_the_generator(tmp_path):
    """docs/figures/report_*.svg is what scripts/report_figures.py produces today (re-run it if this fails)."""
    written = report_figures.write_all(tmp_path)
    committed = Path(report_figures.OUT)
    for p in written:
        assert (committed / p.name).read_text(encoding="utf-8") == p.read_text(encoding="utf-8"), p.name


def test_report_references_every_report_figure():
    text = (Path(report_figures.ROOT) / "docs" / "report_palu_on_tt.md").read_text(encoding="utf-8")
    for name in report_figures.FIGURES:
        assert f"figures/{name}.svg" in text, name
