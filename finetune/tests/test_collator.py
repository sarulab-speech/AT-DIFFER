"""Invariants of the Example, and the padding of the collator.

The boundary of the supervised region is decided in build_example, which
needs the model's processor. These tests check, without the model, the
invariants that do not need it and the construction of the boundary itself.
"""

import pytest

from at_differ_finetune.data.collator import LABEL_IGNORE_INDEX
from at_differ_finetune.data.targets import NO_LINE_ID, build_target
from at_differ_finetune.models.base import Example

torch = pytest.importorskip("torch", reason="the collator needs torch")


def make_example(n_head: int, body_ids: list[int], line_id: int = 3) -> Example:
    """Built like build_example: head and body concatenated, so the boundary is exact."""
    return Example(
        input_ids=[7] * n_head + body_ids,
        labels=[LABEL_IGNORE_INDEX] * n_head + body_ids,
        token_weights=[1.0] * n_head + [1.0] * len(body_ids),
        line_ids=[NO_LINE_ID] * n_head + [line_id] * len(body_ids),
    )


def test_example_rejects_length_mismatch():
    with pytest.raises(ValueError, match="different lengths"):
        Example(input_ids=[1, 2], labels=[1], token_weights=[1.0, 1.0])


def test_only_body_tokens_are_supervised():
    """The number of valid labels equals the body length exactly (no off-by-one)."""
    example = make_example(n_head=5, body_ids=[11, 12, 13])
    supervised = [x for x in example.labels if x != LABEL_IGNORE_INDEX]
    assert supervised == [11, 12, 13]
    # The valid labels are contiguous at the end.
    assert example.labels[-3:] == [11, 12, 13]
    assert set(example.labels[:5]) == {LABEL_IGNORE_INDEX}


def test_supervised_labels_equal_the_input_at_those_positions():
    """Teacher forcing: at supervised positions, labels == input_ids."""
    example = make_example(n_head=4, body_ids=[21, 22])
    for i, label in enumerate(example.labels):
        if label != LABEL_IGNORE_INDEX:
            assert label == example.input_ids[i]


class _StubAdapter:
    def collate_extras(self, batch):
        return {}


def test_collator_pads_each_field_with_its_own_value():
    from at_differ_finetune.data.collator import ExampleCollator

    batch = [make_example(2, [31, 32, 33]), make_example(2, [41])]
    collated = ExampleCollator(adapter=_StubAdapter(), pad_token_id=0)(batch)

    assert collated["input_ids"].shape == (2, 5)
    # The last two positions of the shorter one are padding.
    assert collated["input_ids"][1].tolist() == [7, 7, 41, 0, 0]
    assert collated["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]
    # labels are padded with -100 (excluded from the loss).
    assert collated["labels"][1].tolist() == [-100, -100, 41, -100, -100]
    # token_weights are padded with 0.0.
    assert collated["token_weights"][1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    # line_ids are padded with NO_LINE_ID (matching the -100 labels).
    assert collated["line_ids"][1].tolist() == [0, 0, 3, 0, 0]


def test_line_ids_and_labels_agree_after_padding():
    """The premise of the separate reduction: valid label <=> has a line ID.

    If they disagree, reduce_per_line fails (rather than silently dropping part of a line).
    """
    from at_differ_finetune.data.collator import ExampleCollator

    batch = [make_example(2, [11, 12, 13], line_id=2), make_example(3, [21], line_id=3)]
    collated = ExampleCollator(adapter=_StubAdapter(), pad_token_id=0)(batch)

    supervised = collated["labels"] != LABEL_IGNORE_INDEX
    assert torch.equal(supervised, collated["line_ids"] != NO_LINE_ID)


def test_example_defaults_line_ids_to_no_line():
    """Without line IDs, the lengths still agree."""
    example = Example(input_ids=[1, 2], labels=[1, 2], token_weights=[1.0, 1.0])
    assert example.line_ids == [NO_LINE_ID, NO_LINE_ID]


def test_padding_never_adds_supervised_tokens():
    """Padding does not add supervised tokens (which would train on garbage)."""
    from at_differ_finetune.data.collator import ExampleCollator

    batch = [make_example(2, [1, 2, 3]), make_example(2, [4])]
    collated = ExampleCollator(adapter=_StubAdapter(), pad_token_id=0)(batch)
    supervised = (collated["labels"] != LABEL_IGNORE_INDEX).sum().item()
    assert supervised == 3 + 1


def test_target_prefix_and_body_recompose_exactly(taxonomy):
    """prefix + body is exactly the three lines the model should output."""
    target = build_target(["sound_omission"], "The caption adds the sound of rain.", taxonomy)
    assert target.text.splitlines() == [
        "DISCREPANCY: 1",
        "TYPES: phantom_sound",
        "EXPLANATION: The caption adds the sound of rain.",
    ]
