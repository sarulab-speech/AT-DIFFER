"""Build the three-line target for one metadata row, and choose which lines to supervise.

The model output specified by the prompt (resources/prompt.txt) is::

    DISCREPANCY: <1 or 0>
    TYPES: <comma-separated types, or "none">
    EXPLANATION: <exactly one sentence>

All three lines are placed in the target sequence. Which of them receive a
loss is set by ``supervised_lines`` in the config (see
:func:`split_for_supervision`): Method 1 supervises the EXPLANATION text only,
Methods 2 and 3 supervise all three lines.

The boundary between the unsupervised and the supervised part is not found by
searching the token sequence for "EXPLANATION:", which would depend on how the
tokenizer splits it. Instead, the two parts are kept as separate strings,
tokenized separately, and concatenated, so the boundary is exact by
construction.

The prefix ends with "EXPLANATION:" and the space goes at the start of the
body. BPE merges a space with the following word (" The" is one token), so
putting the space at the end of the prefix would make the concatenated tokens
differ from those of the whole text tokenized at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..taxonomy import Taxonomy


@dataclass(frozen=True)
class Target:
    """A target sequence split into an unsupervised prefix and a supervised body.

    prefix: "DISCREPANCY: 1\\nTYPES: event_omission, phantom_sound\\nEXPLANATION:"
    body:   " The caption mistakes the sound of rain for the sound of applause."
            (the leading space is intentional; see the module docstring)
    """

    prefix: str
    body: str
    #: First line, "DISCREPANCY: 1\n" (with the newline). Kept separately so
    #: that the parts of the prefix can be supervised individually.
    discrepancy_line: str = ""
    #: Second line, "TYPES: ...\n".
    types_line: str = ""
    #: Label of the third line, "EXPLANATION:".
    explanation_label: str = ""

    @property
    def text(self) -> str:
        """The full three lines that the model should output."""
        return self.prefix + self.body


def build_target(
    difference_types: list[str], difference_caption: str, taxonomy: Taxonomy
) -> Target:
    """Build the three-line target from a row's difference types and difference caption.

    The difference types are mapped to the output types of the TYPES line
    with ``taxonomy.to_output_types`` (resources/output_format.yaml, 4.3 in
    the design notes).

    >>> t = build_target(["semantic_difference", "simultaneity_discrepancy"],
    ...                  "The caption mistakes X for Y.", taxonomy)
    >>> t.prefix
    'DISCREPANCY: 1\\nTYPES: event_omission, phantom_sound, overlap\\nEXPLANATION:'
    >>> t.body
    ' The caption mistakes X for Y.'
    """
    explanation = str(difference_caption).strip()
    if not explanation:
        raise ValueError(f"difference_caption is empty (difference_types={difference_types!r})")

    output_types = taxonomy.to_output_types(difference_types)
    has_difference = bool(output_types)

    # A row without a difference must have the fixed explanation, and a row
    # with one must not. Anything else is corrupted data, so fail here instead
    # of training on it.
    is_boilerplate = explanation == taxonomy.no_difference_explanation
    if has_difference == is_boilerplate:
        raise ValueError(
            "difference_types and difference_caption are inconsistent: "
            f"difference_types={difference_types!r} -> types={output_types}, "
            f"difference_caption={explanation!r}"
        )

    discrepancy_key, types_key, explanation_key = taxonomy.keys
    discrepancy_value = (
        taxonomy.discrepancy_with_difference
        if has_difference
        else taxonomy.discrepancy_no_difference
    )
    discrepancy_line = f"{discrepancy_key}: {discrepancy_value}\n"
    types_line = f"{types_key}: {taxonomy.format_types_line(output_types)}\n"
    explanation_label = f"{explanation_key}:"
    return Target(
        prefix=discrepancy_line + types_line + explanation_label,
        body=f" {explanation}",
        discrepancy_line=discrepancy_line,
        types_line=types_line,
        explanation_label=explanation_label,
    )


# --- Choosing the supervised lines -------------------------------------------

#: Line names that can be listed in ``supervised_lines``.
LINE_NAMES = ("discrepancy", "types", "explanation")

#: Default: supervise the EXPLANATION text only (Method 1).
DEFAULT_SUPERVISED_LINES = ("explanation",)

#: Integer IDs of the lines, used by the loss. 0 means "no line", i.e. not
#: supervised. IDs start at 1 so that the head and the padding, which are
#: filled with 0, are "no line" without any extra step.
NO_LINE_ID = 0
LINE_IDS = {name: index for index, name in enumerate(LINE_NAMES, start=1)}


def normalize_supervised_lines(lines) -> tuple[str, ...]:
    """Validate the setting and return it as a tuple in the order of LINE_NAMES."""
    if not lines:
        raise ValueError(f"supervised_lines is empty. Give at least one of {list(LINE_NAMES)}.")
    unknown = sorted(set(lines) - set(LINE_NAMES))
    if unknown:
        raise ValueError(f"Unknown line names {unknown}. Valid names: {list(LINE_NAMES)}.")
    if "explanation" not in lines:
        # Without the third line, the difference caption is not learned at all.
        raise ValueError("supervised_lines must include 'explanation'.")
    return tuple(name for name in LINE_NAMES if name in lines)


def split_for_supervision(
    target: Target, supervised_lines=DEFAULT_SUPERVISED_LINES
) -> tuple[str, list[tuple[str | None, str]]]:
    """Split a target into the part appended to the prompt and the supervised part.

    Returns ``(head_suffix, body_segments)``:

    - head_suffix: string appended after the chat template and the generation
      prefix. It receives no loss.
    - body_segments: ``[(line name or None, text)]``. **A name of None means
      the segment is not supervised**, so callers do not need to check
      ``supervised_lines`` again.

    The rule is: everything before the first supervised line is head, and
    everything from there on is body. None appears only when an unsupervised
    line lies between supervised ones.

    The "EXPLANATION:" label itself:
      - When only the explanation is supervised (Method 1), it is in the
        **head**, i.e. given as part of the prompt.
      - When the first or second line is also supervised (Methods 2 and 3), it
        is **supervised as part of the explanation line**, so that the model
        learns to produce the line structure itself.

    >>> t = build_target(["simultaneity_discrepancy"], "The caption says A and B overlap.", taxonomy)
    >>> head, body = split_for_supervision(t)          # default
    >>> head == t.prefix and body == [("explanation", t.body)]
    True
    >>> head, body = split_for_supervision(t, ("discrepancy", "types", "explanation"))
    >>> head
    ''
    >>> [name for name, _ in body]
    ['discrepancy', 'types', 'explanation', 'explanation']
    """
    supervised = set(supervised_lines)
    structural = bool(supervised & {"discrepancy", "types"})

    pieces: list[tuple[str | None, str]] = [
        ("discrepancy", target.discrepancy_line),
        ("types", target.types_line),
        # Where the label belongs depends on the configuration (see above).
        ("explanation" if structural else None, target.explanation_label),
        ("explanation", target.body),
    ]

    head_parts: list[str] = []
    index = 0
    for name, text in pieces:
        if name in supervised:
            break
        head_parts.append(text)
        index += 1

    body = [
        (name if name in supervised else None, text) for name, text in pieces[index:]
    ]
    return "".join(head_parts), body


def build_supervision(
    token_ids: list[int],
    line_names: list[str | None],
    line_weights: dict[str, float] | None = None,
) -> tuple[list[int], list[float]]:
    """Build (labels, token weights) for the tokens of the body.

    - labels: -100 for tokens of unsupervised lines (line name None).
    - token weights: the weight of the token's line (``line_weights``, 1.0 if
      not given). Tokens of unsupervised lines get 1.0; their weight is never
      used because their label is -100.

    The loss normalizes by the sum of the weights (loss.reduce_per_sample), so
    only the ratio between line weights matters. With no line weights, all
    weights are 1.0 and the balance between lines follows their token counts.

    **With the separate reduction (Method 3), the line weight applied here
    cancels out**, because it appears in both the numerator and the
    denominator of the per-line average. The loss applies it again to the
    line averages, so a line weight takes effect exactly once with either
    reduction, and the same Example serves both (see the loss module).

    >>> build_supervision([5, 6, 7], ["types", "explanation", None])
    ([5, 6, -100], [1.0, 1.0, 1.0])
    >>> build_supervision([5, 6], ["types", "explanation"], {"types": 3.0})
    ([5, 6], [3.0, 1.0])
    """
    if len(token_ids) != len(line_names):
        raise ValueError(f"Length mismatch: {len(token_ids)} / {len(line_names)}")
    weights_by_line = line_weights or {}
    labels = [
        token_id if name is not None else -100
        for token_id, name in zip(token_ids, line_names)
    ]
    weights = [
        weights_by_line.get(name, 1.0) if name is not None else 1.0
        for name in line_names
    ]
    return labels, weights


def encode_line_ids(line_names: list[str | None]) -> list[int]:
    """Convert per-token line names to the integer IDs used by the loss.

    Positions that get NO_LINE_ID here are exactly the positions that
    build_supervision sets to -100, since both are decided by None in the same
    line_names.

    >>> encode_line_ids(["types", None, "explanation"])
    [2, 0, 3]
    """
    return [NO_LINE_ID if name is None else LINE_IDS[name] for name in line_names]


def line_weight_vector(line_weights: dict[str, float] | None = None) -> list[float]:
    """Vector of line weights indexed by line ID (passed to loss.reduce_per_line).

    Index 0 ("no line") is always 0.0. Lines not given get 1.0.

    >>> line_weight_vector()
    [0.0, 1.0, 1.0, 1.0]
    >>> line_weight_vector({"explanation": 2.0})
    [0.0, 1.0, 1.0, 2.0]
    """
    weights = line_weights or {}
    return [0.0] + [float(weights.get(name, 1.0)) for name in LINE_NAMES]


def line_of_offsets(
    offsets: list[tuple[int, int]], body_segments: list[tuple[str | None, str]]
) -> list[str | None]:
    """Return, for each token of the body, the name of its line (None if unsupervised).

    A token belongs to the segment that contains its character offset. A
    token that spans a segment boundary belongs to the segment where it
    **starts**, so every token is assigned even if BPE merges across a
    boundary. Zero-width tokens (such as EOS) belong to the last segment.
    """
    bounds: list[tuple[int, int, str | None]] = []
    cursor = 0
    for name, text in body_segments:
        bounds.append((cursor, cursor + len(text), name))
        cursor += len(text)

    names: list[str | None] = []
    for start, end in offsets:
        if start == end:  # special tokens etc. with no corresponding text
            names.append(bounds[-1][2] if bounds else None)
            continue
        for low, high, name in bounds:
            if low <= start < high:
                names.append(name)
                break
        else:
            names.append(bounds[-1][2] if bounds else None)
    return names
