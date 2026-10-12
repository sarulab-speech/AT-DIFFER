"""Building the three-line target, and mapping difference types to output types."""

import pytest

from at_differ_finetune.data.targets import build_target

CAPTION = "The caption mistakes the sound of rain for the sound of applause."


def test_semantic_difference_is_decomposed(taxonomy):
    """The prompt has no type for a semantic difference: it is an omission plus a phantom sound."""
    assert taxonomy.to_output_types(["semantic_difference"]) == [
        "event_omission",
        "phantom_sound",
    ]


def test_types_are_deduplicated_and_ordered(taxonomy):
    """The TYPES line is a set without multiplicity, in the fixed order of the prompt."""
    assert taxonomy.to_output_types(
        ["semantic_difference", "semantic_difference"]
    ) == ["event_omission", "phantom_sound"]
    # The order of output_types, whatever the input order.
    assert taxonomy.to_output_types(["simultaneity_discrepancy", "sound_omission"]) == [
        "phantom_sound",
        "overlap",
    ]


def test_no_difference_maps_to_nothing(taxonomy):
    assert taxonomy.to_output_types(["no_difference"]) == []
    assert taxonomy.format_types_line([]) == "none"


def test_unknown_type_is_rejected_loudly(taxonomy):
    """A difference type missing from the taxonomy is not let through silently."""
    with pytest.raises(ValueError, match="Unknown difference type"):
        taxonomy.to_output_types(["telepathy"])


@pytest.mark.parametrize(
    "difference_types,types_line",
    [
        # The examples of the design notes (resources, 4.3).
        (["semantic_difference", "simultaneity_discrepancy"], "event_omission, phantom_sound, overlap"),
        (["text_omission", "text_omission"], "event_omission"),
        (["sound_omission", "sequential_discrepancy"], "phantom_sound, swapped_order"),
    ],
)
def test_target_lines_match_the_prompt_format(taxonomy, difference_types, types_line):
    target = build_target(difference_types, CAPTION, taxonomy)
    assert target.prefix == f"DISCREPANCY: 1\nTYPES: {types_line}\nEXPLANATION:"
    assert target.text == target.prefix + target.body
    assert target.text.splitlines() == [
        "DISCREPANCY: 1",
        f"TYPES: {types_line}",
        f"EXPLANATION: {CAPTION}",
    ]


def test_body_starts_with_space_not_prefix(taxonomy):
    """The space goes at the start of the body.

    BPE merges a space with the following word (" The" is one token), so a
    space at the end of the prefix would make the separately tokenized head
    and body differ from the whole text tokenized at once.
    """
    target = build_target(["sound_omission"], CAPTION, taxonomy)
    assert not target.prefix.endswith(" ")
    assert target.body.startswith(" ")
    assert target.body == f" {CAPTION}"


def test_no_difference_target(taxonomy):
    target = build_target(["no_difference"], taxonomy.no_difference_explanation, taxonomy)
    assert target.prefix == "DISCREPANCY: 0\nTYPES: none\nEXPLANATION:"
    assert target.body == " The caption accurately describes the audio."


@pytest.mark.parametrize(
    "difference_types,caption",
    [
        # a difference, but the fixed explanation for no difference
        (["sound_omission"], "The caption accurately describes the audio."),
        # no difference, but a specific difference caption
        (["no_difference"], CAPTION),
    ],
)
def test_inconsistent_rows_are_rejected(taxonomy, difference_types, caption):
    """Disagreeing difference types and difference caption are corrupted data; do not train on them."""
    with pytest.raises(ValueError, match="inconsistent"):
        build_target(difference_types, caption, taxonomy)


def test_empty_caption_is_rejected(taxonomy):
    with pytest.raises(ValueError, match="difference_caption is empty"):
        build_target(["sound_omission"], "   ", taxonomy)
