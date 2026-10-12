"""Which lines receive a loss, and the line weights.

Method 1 supervises the EXPLANATION text only; Methods 2 and 3 supervise all
three lines. The split must be exact for both.
"""

from pathlib import Path

import pytest
import yaml

from at_differ_finetune.config import Config
from at_differ_finetune.data.targets import (
    DEFAULT_SUPERVISED_LINES,
    LINE_NAMES,
    NO_LINE_ID,
    build_supervision,
    build_target,
    encode_line_ids,
    line_of_offsets,
    line_weight_vector,
    normalize_supervised_lines,
    split_for_supervision,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

ALL_LINES = ("discrepancy", "types", "explanation")


@pytest.fixture
def target(taxonomy):
    return build_target(
        ["semantic_difference", "simultaneity_discrepancy"],
        "The caption mistakes the sound of rain for the sound of applause.",
        taxonomy,
    )


# --- Explanation only (Method 1) -----------------------------------------------


def test_default_supervises_the_explanation_text_only(target):
    """By default the head is target.prefix and the body is target.body."""
    head, body = split_for_supervision(target)
    assert head == target.prefix
    assert body == [("explanation", target.body)]


def test_default_is_explanation_only():
    assert DEFAULT_SUPERVISED_LINES == ("explanation",)


def test_split_never_loses_or_duplicates_text(target):
    """In every configuration, head + body is the original target."""
    for lines in (("explanation",), ("types", "explanation"), ALL_LINES):
        head, body = split_for_supervision(target, lines)
        assert head + "".join(text for _, text in body) == target.text


# --- All three lines (Methods 2 and 3) -----------------------------------------


def test_all_lines_moves_everything_into_the_supervised_region(target):
    head, body = split_for_supervision(target, ALL_LINES)
    assert head == ""
    assert [name for name, _ in body] == [
        "discrepancy",
        "types",
        "explanation",  # the "EXPLANATION:" label
        "explanation",  # the text
    ]


def test_explanation_label_follows_the_structure_decision(target):
    """"EXPLANATION:" is supervised only when the line structure is taught."""
    _, default_body = split_for_supervision(target, ("explanation",))
    assert all("EXPLANATION:" not in text for _, text in default_body)

    _, all_body = split_for_supervision(target, ALL_LINES)
    labelled = [name for name, text in all_body if text == "EXPLANATION:"]
    assert labelled == ["explanation"]


def test_dropping_only_the_first_line(target):
    """Without the first line, only that line goes to the head."""
    head, body = split_for_supervision(target, ("types", "explanation"))
    assert head == target.discrepancy_line
    assert [name for name, _ in body] == ["types", "explanation", "explanation"]


def test_skipping_a_middle_line_marks_it_unsupervised(target):
    """Skipping the second line leaves an unsupervised segment (None) inside the body."""
    head, body = split_for_supervision(target, ("discrepancy", "explanation"))
    assert head == ""
    names = [name for name, _ in body]
    assert names[0] == "discrepancy"
    assert None in names, "the unsupervised TYPES line should remain as None"


# --- Validating the settings --------------------------------------------------


def test_unknown_line_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown line names"):
        normalize_supervised_lines(["explanation", "typo"])


def test_explanation_cannot_be_dropped():
    """Without the third line, the difference caption would not be learned."""
    with pytest.raises(ValueError, match="explanation"):
        normalize_supervised_lines(["discrepancy", "types"])


def test_empty_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        normalize_supervised_lines([])


def test_order_is_normalized():
    assert normalize_supervised_lines(["explanation", "discrepancy"]) == (
        "discrepancy",
        "explanation",
    )


def _config(**overrides):
    config = Config(train_manifest="x.jsonl", output_dir="out", **overrides)
    config.validate()
    return config


def test_loss_reduction_defaults_to_concatenate():
    assert _config().loss_reduction == "concatenate"


def test_unknown_loss_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unknown loss_reduction"):
        _config(loss_reduction="per_token")


def test_separate_needs_more_than_one_supervised_line():
    """With one line, separate equals concatenate, so the setting would have no effect."""
    with pytest.raises(ValueError, match="a single line"):
        _config(loss_reduction="separate")

    _config(loss_reduction="separate", supervised_lines=list(ALL_LINES))  # this is fine


def test_line_weight_on_an_unsupervised_line_is_rejected():
    with pytest.raises(ValueError, match="not in supervised_lines"):
        _config(line_weights={"types": 2.0})


def test_unknown_config_key_is_rejected(tmp_path):
    """A typo in a setting name fails instead of being ignored."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"train_manifest": "x.jsonl", "output_dir": "out", "lerning_rate": 1e-4}))
    with pytest.raises(KeyError, match="lerning_rate"):
        Config.load(path)


def test_model_path_is_made_absolute(tmp_path, monkeypatch):
    """model_path is recorded in adapter_config.json and used by `merge`, so it
    must not depend on the directory `merge` is run from."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"train_manifest": "x.jsonl", "output_dir": "out", "model_path": "../models/base"})
    )
    config = Config.load(path)
    assert config.model_path == str((tmp_path / "../models/base").resolve())


# --- Assigning tokens to lines ------------------------------------------------


def test_offsets_map_to_the_right_lines():
    segments = [("discrepancy", "AAA"), ("types", "BBBB"), ("explanation", "CC")]
    #            0..2               3..6             7..8
    offsets = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 9)]
    assert line_of_offsets(offsets, segments) == [
        "discrepancy",
        "discrepancy",
        "types",
        "types",
        "explanation",
    ]


def test_token_spanning_a_boundary_belongs_to_where_it_starts():
    """Every token is assigned even if BPE merges across a boundary."""
    segments = [("types", "AAAA"), ("explanation", "BBBB")]
    offsets = [(2, 6)]  # from the middle of types to the middle of explanation
    assert line_of_offsets(offsets, segments) == ["types"]


def test_zero_width_token_is_attributed_to_the_last_segment():
    """Tokens with no text, such as EOS, belong to the last line."""
    segments = [("discrepancy", "AAA"), ("explanation", "BB")]
    assert line_of_offsets([(0, 0)], segments) == ["explanation"]


def test_unsupervised_segment_is_reported_as_none():
    segments = [("discrepancy", "AAA"), (None, "XX"), ("explanation", "BB")]
    assert line_of_offsets([(0, 1), (3, 4), (5, 6)], segments) == [
        "discrepancy",
        None,
        "explanation",
    ]


def test_every_line_name_is_covered_by_the_constant():
    assert set(ALL_LINES) == set(LINE_NAMES)


# --- Labels and weights -------------------------------------------------------


def test_unsupervised_tokens_are_masked():
    labels, _ = build_supervision([11, 12, 13], ["types", None, "explanation"])
    assert labels == [11, -100, 13]


def test_line_weights_become_token_weights():
    _, weights = build_supervision(
        [1, 2, 3], ["types", "explanation", "explanation"], {"explanation": 2.0}
    )
    assert weights == [1.0, 2.0, 2.0]


def test_no_line_weights_give_weight_one():
    """Without line weights (the paper's runs), every weight is 1.0."""
    _, weights = build_supervision([1, 2, 3], ["discrepancy", "types", "explanation"], {})
    assert weights == [1.0, 1.0, 1.0]


def test_unsupervised_tokens_keep_weight_one():
    """The weight at -100 positions is unused, but kept at 1.0."""
    _, weights = build_supervision([1, 2], ["explanation", None], {"explanation": 3.0})
    assert weights == [3.0, 1.0]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Length mismatch"):
        build_supervision([1, 2], ["explanation"])


def test_relative_weights_are_what_matter():
    """Scaling all line weights by a constant gives the same normalized weights."""
    ids, names = [1, 2, 3], ["types", "explanation", "explanation"]
    _, a = build_supervision(ids, names, {"types": 2.0, "explanation": 1.0})
    _, b = build_supervision(ids, names, {"types": 4.0, "explanation": 2.0})
    assert [w / sum(a) for w in a] == [w / sum(b) for w in b]


# --- Line IDs (read by the separate reduction) --------------------------------


def test_line_ids_follow_the_declared_order():
    assert encode_line_ids(list(LINE_NAMES)) == [1, 2, 3]


def test_unsupervised_tokens_get_no_line_id():
    assert encode_line_ids(["types", None, "explanation"]) == [2, NO_LINE_ID, 3]


def test_line_ids_and_labels_mask_exactly_the_same_positions():
    """**If these disagree, part of a line is silently left untrained.**

    reduce_per_line rejects "label not -100 but line ID 0"; here the premise
    (both come from the same line_names) is checked on the pure functions.
    """
    names = ["discrepancy", None, "types", "explanation", None]
    labels, _ = build_supervision([1, 2, 3, 4, 5], names)
    line_ids = encode_line_ids(names)
    assert [label == -100 for label in labels] == [i == NO_LINE_ID for i in line_ids]


def test_line_weight_vector_is_indexed_by_line_id():
    """The λ vector can be indexed by the IDs of encode_line_ids."""
    lam = line_weight_vector({"types": 2.0})
    assert lam[NO_LINE_ID] == 0.0  # "no line" is not in the loss
    for name, expected in (("discrepancy", 1.0), ("types", 2.0), ("explanation", 1.0)):
        assert lam[encode_line_ids([name])[0]] == expected


# --- The configs of the paper -------------------------------------------------


@pytest.mark.parametrize(
    "base_name,other_name,axis",
    [
        # Method 1 vs 2: which lines are supervised.
        ("method1", "method2", {"supervised_lines"}),
        # Method 2 vs 3: how the loss is reduced (both supervise all three lines).
        ("method2", "method3", {"loss_reduction"}),
    ],
)
def test_configs_differ_only_on_the_axis_under_test(base_name, other_name, axis):
    """The method configs differ only in the intended setting and output_dir."""
    base = Config.load(CONFIGS_DIR / f"{base_name}.yaml")
    other = Config.load(CONFIGS_DIR / f"{other_name}.yaml")
    differing = {name for name in vars(base) if getattr(base, name) != getattr(other, name)}
    assert differing == axis | {"output_dir"}
