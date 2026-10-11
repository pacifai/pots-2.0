"""EV1: figures as PDF and PNG, tables as CSV and Markdown, under evaluation/results/<scale>/."""

import csv
import math

import pytest

from evaluation import output as eo


def test_results_dir_is_under_the_output_dir(tmp_path, monkeypatch):
    assert eo.results_dir("test", root=tmp_path) == tmp_path / "evaluation" / "results" / "test"
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path / "out"))
    assert eo.results_dir("full") == tmp_path / "out" / "evaluation" / "results" / "full"
    with pytest.raises(ValueError, match="scale must be one of"):
        eo.results_dir("medium", root=tmp_path)


@pytest.mark.parametrize("name", ["", ".hidden", "a/b", "../up", "a\\b"])
def test_names_are_plain_file_names(tmp_path, name):
    with pytest.raises(ValueError, match="plain file name"):
        eo.write_table([{"a": 1}], name, "test", root=tmp_path)


def test_figure_written_as_pdf_and_png(tmp_path):
    fig, (left, right) = eo.new_figure(1, 2, figsize=(6, 3))
    left.plot([1, 2, 3], [1, 4, 9])
    right.bar(["a", "b"], [1, 2])
    pdf, png = eo.write_figure(fig, "detection", "test", root=tmp_path)
    d = tmp_path / "evaluation" / "results" / "test"
    assert (pdf, png) == (d / "detection.pdf", d / "detection.png")
    assert pdf.read_bytes().startswith(b"%PDF") and png.read_bytes().startswith(b"\x89PNG")
    first = pdf.read_bytes()
    eo.write_figure(fig, "detection", "test", root=tmp_path)
    assert pdf.read_bytes() == first  # no creation date: the same figure gives the same file


def test_table_written_as_csv_and_markdown(tmp_path):
    rows = [{"model": "m|1", "rate": 0.1 + 0.2, "n": 3, "note": None},
            {"model": "m2", "rate": math.nan, "n": 12, "extra": [1, 2]}]
    csv_path, md_path = eo.write_table(rows, "rates", "full", root=tmp_path,
                                       formats={"rate": ".2f"}, caption="Mean over 3 seeds.")
    d = tmp_path / "evaluation" / "results" / "full"
    assert (csv_path, md_path) == (d / "rates.csv", d / "rates.md")
    with open(csv_path, newline="") as f:
        back = list(csv.reader(f))
    assert back == [["model", "rate", "n", "note", "extra"],
                    ["m|1", "0.30000000000000004", "3", "", ""],
                    ["m2", "nan", "12", "", "[1, 2]"]]
    assert md_path.read_text() == (
        "Mean over 3 seeds.\n"
        "\n"
        "| model | rate | n | note | extra |\n"
        "|---|---|---|---|---|\n"
        "| m\\|1 | 0.30 | 3 |  |  |\n"
        "| m2 | nan | 12 |  | [1, 2] |\n")


def test_table_columns_pick_and_order(tmp_path):
    rows = [{"a": 1.23456789, "b": "x", "c": 3}]
    csv_path, md_path = eo.write_table(rows, "t", "test", root=tmp_path, columns=["c", "a"])
    assert csv_path.read_text() == "c,a\n3,1.23456789\n"
    assert md_path.read_text() == "| c | a |\n|---|---|\n| 3 | 1.235 |\n"
    with pytest.raises(ValueError, match="no columns"):
        eo.write_table([], "empty", "test", root=tmp_path)
    _, md_path = eo.write_table([], "header", "test", root=tmp_path, columns=["a"])
    assert md_path.read_text() == "| a |\n|---|\n"
