"""scripts/eval_live_labels.py on an invented labelled CSV."""
import csv
import json
import os

import pytest

from scripts.eval_live_labels import NOTE, evaluate, main, read_labels
from src.common import config

REFERENCE = {"macro_f1": 0.7, "macro_f1_ci95": [0.68, 0.72]}


def write_labels(path, rows, header=("sample_id", "text", "label")):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def invented_rows():
    rows = []
    for i in range(10):
        rows.append([f"n{i}", f"this is terrible and awful {i}", "negative"])
        rows.append([f"u{i}", f"the meeting is at noon {i}", "neutral"])
        rows.append([f"p{i}", f"what a wonderful lovely day {i}", "Positive "])  # case and spaces are tidied
    rows += [["s1", "not sure what this is", "skip"], ["e1", "nobody labelled this", ""]]
    return rows


def keyword_model(texts):
    """A stand-in model: right on every invented row except the first neutral one."""
    out = []
    for t in texts:
        if "terrible" in t:
            out.append("negative")
        elif "wonderful" in t:
            out.append("positive")
        else:
            out.append("negative" if t.endswith(" 0") else "neutral")
    return out


def test_skip_and_empty_rows_are_not_scored(tmp_path):
    scored, counts = read_labels(write_labels(tmp_path / "labels.csv", invented_rows()))
    assert counts == {"rows": 32, "skip": 1, "empty": 1, "scored": 30}
    assert set(scored["label"]) == {"negative", "neutral", "positive"}


def test_unknown_label_or_missing_column_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        read_labels(write_labels(tmp_path / "a.csv", [["x", "some text", "angry"]]))
    with pytest.raises(SystemExit):
        read_labels(write_labels(tmp_path / "b.csv", [["x", "some text"]], header=("sample_id", "text")))
    with pytest.raises(SystemExit):
        read_labels(write_labels(tmp_path / "c.csv", [["x", "some text", "skip"]]))  # nothing to score


def test_refuses_a_labels_file_inside_the_repository(tmp_path):
    inside = os.path.join(config.BASE_DIR, "data", "labels.csv")
    with pytest.raises(SystemExit):
        read_labels(inside)


def test_scores_baseline_and_comparison(tmp_path):
    scored, _ = read_labels(write_labels(tmp_path / "labels.csv", invented_rows()))
    result = evaluate(scored, keyword_model, REFERENCE)
    model = result["model"]
    assert model["accuracy"] == pytest.approx(29 / 30, abs=1e-4)
    assert model["confusion_matrix"] == [[10, 0, 0], [1, 9, 0], [0, 0, 10]]
    assert model["f1_per_class"]["positive"] == 1.0
    low, high = model["macro_f1_ci95"]
    assert low <= model["macro_f1"] <= high <= 1.0
    # three labels with 10 rows each: the tie goes to the first label alphabetically
    assert result["baseline"]["predicts"] == "negative"
    assert result["baseline"]["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["macro_f1_difference_vs_baseline"]["value"] > 0
    shift = result["vs_2015_test_set"]
    assert shift["macro_f1_live_minus_2015"] == round(model["macro_f1"] - 0.7, 4)
    assert shift["cis_overlap"] is False
    assert evaluate(scored, keyword_model, REFERENCE) == result  # the bootstrap is seeded


def test_main_writes_aggregates_only(tmp_path):
    labels = write_labels(tmp_path / "labels.csv", invented_rows())
    output = tmp_path / "live_domain_shift.json"
    assert main(["--labels", labels, "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["note"] == NOTE and "Single annotator" in report["note"]
    assert report["n"] == 30 and report["rows"]["skip"] == 1
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        assert report["vs_2015_test_set"]["macro_f1_2015"] == json.load(f)["test"]["model"]["macro_f1"]
    blob = json.dumps(report)
    assert "wonderful" not in blob and "terrible" not in blob and '"n0"' not in blob
