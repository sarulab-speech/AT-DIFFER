"""The design decisions of the loss.

  (1) Per-sample normalization: a sample counts the same however long its caption is.
  (2) Token weights only redistribute the loss within a sample: they do not
      change how much a sample contributes in total, so (1) still holds.
  (3) The reductions behave as intended:
      concatenate = the supervised lines as one text (shares follow token counts)
      separate    = each line normalized first (line length has no effect)
  (4) λ (line_weights) takes effect exactly once with either reduction, so the
      same Example serves both (see the loss module docstring).
"""

import pytest

torch = pytest.importorskip("torch", reason="the loss needs torch")

from at_differ_finetune.loss import (  # noqa: E402
    CONCATENATE,
    LABEL_IGNORE_INDEX,
    SEPARATE,
    build_loss_fn,
    concatenate_loss,
    reduce_per_line,
    reduce_per_sample,
    separate_loss,
    token_cross_entropy,
)


def _reference_reduce(token_loss, mask, weights):
    """A naive reference implementation of the same formula, for comparison."""
    per_sample = []
    for row_loss, row_mask, row_weight in zip(token_loss, mask, weights):
        numerator = sum(l * m * w for l, m, w in zip(row_loss, row_mask, row_weight))
        denominator = sum(m * w for m, w in zip(row_mask, row_weight))
        if denominator > 0:
            per_sample.append(numerator / denominator)
    return sum(per_sample) / len(per_sample)


def test_reduce_matches_a_naive_reference():
    token_loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    weights_full = torch.tensor([[9.0, 1.0, 1.0, 2.0], [9.0, 1.0, 1.0, 1.0]])

    got = reduce_per_sample(token_loss, valid, weights_full).item()
    # weights have the length of input_ids and are shifted by [:, 1:] inside
    expected = _reference_reduce(
        token_loss.tolist(),
        valid.float().tolist(),
        weights_full[:, 1:].tolist(),
    )
    assert got == pytest.approx(expected, rel=1e-6)


def test_long_and_short_samples_contribute_equally():
    """The core of per-sample normalization.

    Sample 1 has one valid token with loss 4.0; sample 2 has four valid tokens,
    all with loss 1.0. The token mean would be (4 + 1*4) / 5 = 1.6; the sample
    mean is (4 + 1) / 2 = 2.5.
    """
    token_loss = torch.tensor([[4.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    valid = torch.tensor(
        [[True, False, False, False], [True, True, True, True]]
    )
    got = reduce_per_sample(token_loss, valid).item()
    assert got == pytest.approx(2.5)

    token_mean = (token_loss * valid).sum() / valid.sum()
    assert token_mean.item() == pytest.approx(1.6)
    assert got != pytest.approx(token_mean.item())


def test_uniform_token_weights_change_nothing():
    """The same weight on every token gives the same result (the denominator is the weight sum)."""
    token_loss = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    valid = torch.tensor([[True, True], [True, True]])
    ones = torch.ones(2, 3)

    unweighted = reduce_per_sample(token_loss, valid).item()
    uniform = reduce_per_sample(token_loss, valid, ones * 2.5).item()
    assert uniform == pytest.approx(unweighted)


def test_token_weights_redistribute_within_a_sample():
    """Weights change the balance within a sample, towards the heavier tokens."""
    token_loss = torch.tensor([[1.0, 5.0]])  # the second token is weighted more
    valid = torch.tensor([[True, True]])
    weights = torch.tensor([[1.0, 1.0, 3.0]])  # [1.0, 3.0] after the shift

    flat = reduce_per_sample(token_loss, valid).item()
    weighted = reduce_per_sample(token_loss, valid, weights).item()
    assert flat == pytest.approx(3.0)  # (1 + 5) / 2
    assert weighted == pytest.approx(4.0)  # (1*1 + 5*3) / (1 + 3)
    assert weighted > flat


def test_token_weights_do_not_inflate_a_samples_total_contribution():
    """A sample with heavier tokens does not get a larger gradient overall.

    This would break if the denominator were the number of tokens.
    """
    # Both samples have loss 2.0 on every token; one has heavy tokens, one does not.
    token_loss = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    valid = torch.tensor([[True, True], [True, True]])
    heavy = torch.tensor([[1.0, 5.0, 5.0], [1.0, 1.0, 1.0]])

    assert reduce_per_sample(token_loss, valid, heavy).item() == pytest.approx(2.0)
    assert reduce_per_sample(token_loss, valid).item() == pytest.approx(2.0)


def test_ignored_positions_do_not_contribute():
    """Positions labelled -100 (unsupervised lines and padding) are not in the loss."""
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.full((2, 6), LABEL_IGNORE_INDEX)
    labels[:, -2:] = torch.tensor([[3, 4], [5, 6]])

    _, valid = token_cross_entropy(logits, labels)
    assert valid.sum().item() == 4  # shifted by one: 2 samples x 2 tokens

    # Changing the logits at ignored positions does not change the loss.
    before = concatenate_loss(logits, labels).item()
    logits[:, :3, :] += 100.0
    after = concatenate_loss(logits, labels).item()
    assert after == pytest.approx(before, rel=1e-5)


def test_loss_drops_when_the_target_token_is_favoured():
    """Sanity check: raising the logit of the correct token lowers the loss."""
    logits = torch.zeros(1, 3, 8)
    labels = torch.tensor([[LABEL_IGNORE_INDEX, 5, 6]])

    baseline = concatenate_loss(logits, labels).item()
    logits[0, 1, 6] = 10.0  # position 1 predicts the label at position 2 (6)
    assert concatenate_loss(logits, labels).item() < baseline


# --- Per-line normalization (loss_reduction: separate) ------------------------
#
# The examples below have two lines, ID 2 (types) and ID 3 (explanation).
# line_ids and weights have the length of input_ids (the loss shifts them by
# [:, 1:]), so a dummy element is prepended.

EQUAL = [0.0, 1.0, 1.0, 1.0]  # λ treating the three lines equally (empty line_weights)


def _shifted(rows):
    """A tensor that becomes ``rows`` after [:, 1:] (a dummy element prepended)."""
    return torch.tensor([[row[0], *row] for row in rows])


def test_separate_normalizes_each_line_before_combining():
    """The share of a line is set by the number of lines, not their length.

    types has one token with loss 5.0; explanation has three tokens with loss 1.0.
      concatenate … (5 + 1 + 1 + 1) / 4 = 2.0   <- pulled towards the long explanation
      separate    … (5 + 1) / 2         = 3.0   <- the two lines count equally
    """
    token_loss = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    line_ids = _shifted([[2, 3, 3, 3]])

    assert reduce_per_sample(token_loss, valid).item() == pytest.approx(2.0)
    assert reduce_per_line(token_loss, valid, line_ids, EQUAL).item() == pytest.approx(3.0)


def test_line_length_does_not_change_the_balance():
    """Making the explanation three times longer does not change separate (it changes concatenate)."""
    short_loss = torch.tensor([[5.0, 1.0]])
    short_ids = _shifted([[2, 3]])
    long_loss = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
    long_ids = _shifted([[2, 3, 3, 3]])
    short_valid, long_valid = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)

    short = reduce_per_line(short_loss, short_valid, short_ids, EQUAL).item()
    long = reduce_per_line(long_loss, long_valid, long_ids, EQUAL).item()
    assert short == pytest.approx(long) == pytest.approx(3.0)

    # With concatenate, the length sets the balance.
    assert reduce_per_sample(short_loss, short_valid).item() == pytest.approx(3.0)
    assert reduce_per_sample(long_loss, long_valid).item() == pytest.approx(2.0)


def test_lambda_shifts_the_balance_between_lines():
    """With separate, line_weights apply directly to the line averages."""
    token_loss = torch.tensor([[5.0, 1.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    line_ids = _shifted([[2, 3]])

    # explanation x3: (1*5 + 3*1) / (1 + 3) = 2.0
    lam = [0.0, 1.0, 1.0, 3.0]
    assert reduce_per_line(token_loss, valid, line_ids, lam).item() == pytest.approx(2.0)


def test_only_the_ratio_of_lambda_matters():
    """Scaling all λ by a constant changes nothing (the denominator is the λ sum)."""
    token_loss = torch.tensor([[5.0, 1.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    line_ids = _shifted([[2, 3]])

    a = reduce_per_line(token_loss, valid, line_ids, [0.0, 1.0, 1.0, 3.0]).item()
    b = reduce_per_line(token_loss, valid, line_ids, [0.0, 10.0, 10.0, 30.0]).item()
    assert a == pytest.approx(b)


def test_lambda_baked_into_token_weights_cancels_out():
    """**The invariant that makes the Example independent of the reduction.**

    build_supervision multiplies λ into the token weights (for concatenate).
    With separate, it cancels out in the per-line weighted average, and λ
    takes effect once, in reduce_per_line.
    """
    token_loss = torch.tensor([[5.0, 1.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    line_ids = _shifted([[2, 3]])
    lam = [0.0, 1.0, 1.0, 3.0]
    # The weights build_supervision makes: 1.0 on types, 3.0 on explanation.
    baked = _shifted([[1.0, 3.0]])

    without = reduce_per_line(token_loss, valid, line_ids, lam).item()
    with_baked = reduce_per_line(token_loss, valid, line_ids, lam, baked).item()
    assert with_baked == pytest.approx(without)


def test_absent_lines_are_excluded_from_the_denominator():
    """A sample missing a line is normalized over the remaining lines (the missing one is not a zero loss)."""
    token_loss = torch.tensor([[4.0, 0.0]])
    valid = torch.tensor([[True, False]])
    line_ids = _shifted([[3, 0]])  # explanation only

    got = reduce_per_line(token_loss, valid, line_ids, EQUAL).item()
    assert got == pytest.approx(4.0)  # not divided by three lines (1.33)


def test_a_single_line_matches_concatenate():
    """With one line, separate equals concatenate (the formulas coincide)."""
    token_loss = torch.tensor([[1.0, 3.0, 5.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    line_ids = _shifted([[3, 3, 3]])

    assert reduce_per_line(token_loss, valid, line_ids, EQUAL).item() == pytest.approx(
        reduce_per_sample(token_loss, valid).item()
    )


def test_supervised_token_without_a_line_id_is_rejected():
    """If labels and line_ids disagree, fail instead of silently dropping part of a line."""
    token_loss = torch.tensor([[1.0, 2.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    line_ids = _shifted([[3, 0]])  # the second token is supervised but has no line ID

    with pytest.raises(ValueError, match="line_id"):
        reduce_per_line(token_loss, valid, line_ids, EQUAL)


# --- Choosing the reduction ---------------------------------------------------


def test_build_loss_fn_dispatches_and_passes_lambda():
    logits = torch.zeros(1, 4, 8)
    labels = torch.tensor([[LABEL_IGNORE_INDEX, 2, 3, 4]])
    line_ids = torch.tensor([[0, 2, 3, 3]])

    concatenate = build_loss_fn(CONCATENATE)
    separate = build_loss_fn(SEPARATE, EQUAL)
    # Uniform logits give every token the same loss, so both reductions agree.
    assert concatenate(logits, labels).item() == pytest.approx(
        separate(logits, labels, line_ids=line_ids).item()
    )


def test_concatenate_ignores_line_inputs():
    """Passing line_ids to concatenate changes nothing (the same Example serves both)."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[LABEL_IGNORE_INDEX, 2, 3, 4]])
    line_ids = torch.tensor([[0, 2, 3, 3]])

    assert concatenate_loss(logits, labels).item() == pytest.approx(
        concatenate_loss(logits, labels, line_ids=line_ids, line_lambda=EQUAL).item()
    )


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unknown loss_reduction"):
        build_loss_fn("per_token")


def test_separate_requires_lambda_at_build_time():
    """Configuration errors surface when the loss is built, not at the first step on the GPUs."""
    with pytest.raises(ValueError, match="line_lambda"):
        build_loss_fn(SEPARATE)


def test_separate_requires_line_ids_at_call_time():
    logits = torch.zeros(1, 3, 8)
    labels = torch.tensor([[LABEL_IGNORE_INDEX, 5, 6]])
    with pytest.raises(ValueError, match="line_ids"):
        separate_loss(logits, labels, line_lambda=EQUAL)
