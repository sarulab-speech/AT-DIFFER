"""Cross-entropy over the supervised tokens, normalized per sample and weighted.

Why plain cross-entropy
-----------------------
The reference difference captions are generated from templates: a small set
of sentence patterns whose slots are filled with FSD50K class names. Since
the references are deterministic, there is no reason to optimize a caption
metric directly with reinforcement learning. The evaluation metrics (BLEU,
CIDEr-D, SPIDEr, BERTScore, ...) all reward agreement with the reference,
which is what minimizing cross-entropy on the reference does.

Per-sample normalization
------------------------
The default cross-entropy averages over all valid tokens in the batch.
Difference captions range from about 16 words (median) to 42 words, so a long
caption would weigh two to three times as much as a short one. Long captions
are mostly those with two differences, so this would amount to weighting by
the number of differences. Instead, the loss is averaged within each sample
and then across samples, so every sample counts equally.

Token weights
-------------
Each token carries a weight: the weight of its line (``line_weights``, all 1
in the paper's runs; see data.targets.build_supervision). The average within
a sample is weighted, and **its denominator is the sum of the weights, not the
number of tokens**, so weights only redistribute the loss within a sample and
never change how much a sample contributes in total.

Reductions (``loss_reduction``)
-------------------------------
When all three lines are supervised, the supervised tokens can be treated as
one text or as three tasks:

    concatenate  L = mean_sample[ Σ_t w_t·l_t / Σ_t w_t ]                         (Methods 1, 2)
    separate     L = mean_sample[ Σ_ℓ λ_ℓ·(Σ_{t∈ℓ} w_t·l_t / Σ_{t∈ℓ} w_t) / Σ_ℓ λ_ℓ ]   (Method 3)

``concatenate`` averages all supervised tokens together, so **the share of
each line is set by its number of tokens**. On AT-DIFFER this gives about 8%
to DISCREPANCY, 7-26% to TYPES (depending on how many types apply), and
61-80% to EXPLANATION.
``separate`` normalizes each line first, so the length of a line has no
effect and the share of each line is set by λ (``line_weights``) alone.

Why λ takes effect only once
----------------------------
data.targets.build_supervision already multiplies λ into the token weights
w_t. Under ``separate``, λ is constant within a line, so it cancels out in the
per-line weighted average. reduce_per_line applies it again to the line
averages, so λ takes effect exactly once with either reduction, and the same
Example serves both (tests/test_loss.py checks this).
"""

from __future__ import annotations

import functools
from typing import Any

LABEL_IGNORE_INDEX = -100

#: Values of ``loss_reduction``. Checked by config.Config.validate.
CONCATENATE = "concatenate"
SEPARATE = "separate"


def token_cross_entropy(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Return the next-token cross-entropy of each token.

    In a causal LM, the logits at position t predict the label at t+1, so the
    two are shifted by one here.

    Returns ``(token_loss, valid_mask)``, both of shape (batch, seq_len - 1).
    Invalid positions of token_loss are 0, so it can be summed directly.

    **Only the supervised positions are cast to fp32.** Casting all positions
    would take about 1.7 GB for the fp32 copy alone (sequence length 2,800 ×
    vocabulary 151,936), and 4-5 GB per sample together with what log_softmax
    keeps for the backward pass. Only a few dozen tokens per sample are
    supervised (the prompt of about 2,500 tokens is -100), so this saves two
    orders of magnitude. The values are the same as without the selection,
    because cross_entropy with reduction="none" treats positions independently.
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid_mask = shift_labels != LABEL_IGNORE_INDEX

    token_loss = torch.zeros(
        shift_labels.shape, dtype=torch.float32, device=shift_logits.device
    )
    if not bool(valid_mask.any()):
        # Add a zero derived from the logits to keep the graph connected.
        return token_loss + shift_logits.sum() * 0.0, valid_mask

    # The boolean mask flattens the first two dimensions: (valid tokens, vocabulary).
    selected_loss = F.cross_entropy(
        shift_logits[valid_mask].float(),
        shift_labels[valid_mask],
        reduction="none",
    )
    token_loss = token_loss.masked_scatter(valid_mask, selected_loss)

    return token_loss, valid_mask


def reduce_per_sample(
    token_loss: Any, valid_mask: Any, weights: Any | None = None, eps: float = 1e-8
) -> Any:
    """Reduce per-token losses: weighted average within each sample, then mean over samples.

    The denominator is the sum of the weights, so weights only redistribute
    the loss within a sample.

    A sample with no valid tokens (which should not happen, but could if a
    row of padding slipped in) has a zero denominator and is left out of the
    mean.
    """
    mask = valid_mask.float()
    if weights is not None:
        # weights have the length of input_ids; shift them like token_loss.
        mask = mask * weights[:, 1:].to(mask.dtype)

    numerator = (token_loss * mask).sum(dim=1)
    denominator = mask.sum(dim=1)

    has_tokens = denominator > 0
    if not bool(has_tokens.any()):
        # Every sample is empty. Return a zero derived from token_loss to keep the graph.
        return token_loss.sum() * 0.0

    per_sample = numerator[has_tokens] / denominator[has_tokens].clamp(min=eps)
    return per_sample.mean()


def reduce_per_line(
    token_loss: Any,
    valid_mask: Any,
    line_ids: Any,
    line_lambda: list[float],
    weights: Any | None = None,
    eps: float = 1e-8,
) -> Any:
    """Reduce per-token losses: average within each line, weighted average over lines, mean over samples.

    Compared with reduce_per_sample, there is one more level of
    normalization. The length of a line no longer affects its share, which is
    set by λ (``line_lambda``) alone.

    ``line_ids`` has the length of input_ids and holds the line IDs of
    data.targets.LINE_IDS (0 = no line); it is shifted like token_loss.
    ``line_lambda`` is the vector of λ indexed by line ID, made by
    data.targets.line_weight_vector (index 0 is always 0.0, so "no line" drops
    out).

    The λ denominator is the sum of λ over the lines **present in the sample**,
    so, as in reduce_per_sample, only the ratio of the λ matters. If a line is
    absent from a sample, the remaining lines are renormalized; the absent
    line is not counted as a zero loss.
    """
    import torch

    ids = line_ids[:, 1:]
    mask = valid_mask.float()
    if weights is not None:
        mask = mask * weights[:, 1:].to(mask.dtype)

    # A supervised token without a line ID would silently leave part of a line
    # untrained. This cannot happen if the Example is built correctly, so it
    # is checked here only to catch bugs.
    if bool((valid_mask & (ids == 0)).any()):
        raise ValueError(
            "Some tokens have a label other than -100 but line_id 0. The line_ids and"
            " labels of the Example disagree (see data.targets.encode_line_ids and"
            " build_supervision)."
        )

    numerators, denominators = [], []
    for line_id, lam in enumerate(line_lambda):
        if line_id == 0 or lam <= 0:
            continue
        line_mask = mask * (ids == line_id)
        line_total = line_mask.sum(dim=1)
        # Count λ in the denominator only if the line is present in the sample.
        present = (line_total > 0).to(mask.dtype)
        line_mean = (token_loss * line_mask).sum(dim=1) / line_total.clamp(min=eps)
        numerators.append(lam * present * line_mean)
        denominators.append(lam * present)

    if not numerators:
        raise ValueError(f"No line has a positive λ: line_lambda={line_lambda}")

    numerator = torch.stack(numerators).sum(dim=0)
    denominator = torch.stack(denominators).sum(dim=0)

    has_lines = denominator > 0
    if not bool(has_lines.any()):
        # Every sample is empty. Return a zero derived from token_loss to keep the graph.
        return token_loss.sum() * 0.0

    return (numerator[has_lines] / denominator[has_lines].clamp(min=eps)).mean()


# --- Choosing the reduction ---------------------------------------------------
#
# Both functions below have the same signature:
#   fn(logits, labels, token_weights=None, line_ids=None, line_lambda=None) -> scalar


def concatenate_loss(
    logits: Any,
    labels: Any,
    token_weights: Any | None = None,
    line_ids: Any | None = None,
    line_lambda: list[float] | None = None,
) -> Any:
    """Average all supervised tokens together, regardless of their line (Methods 1, 2).

    The collator has already set every unsupervised label to -100, so the
    valid tokens here are exactly the supervised ones. line_ids and
    line_lambda are not used.
    """
    token_loss, valid_mask = token_cross_entropy(logits, labels)
    return reduce_per_sample(token_loss, valid_mask, token_weights)


def separate_loss(
    logits: Any,
    labels: Any,
    token_weights: Any | None = None,
    line_ids: Any | None = None,
    line_lambda: list[float] | None = None,
) -> Any:
    """Normalize each line, then take the λ-weighted average of the lines (Method 3)."""
    if line_ids is None or line_lambda is None:
        raise ValueError(
            "The separate loss needs line_ids and line_lambda. Build it with build_loss_fn."
        )
    token_loss, valid_mask = token_cross_entropy(logits, labels)
    return reduce_per_line(token_loss, valid_mask, line_ids, line_lambda, token_weights)


LOSS_FUNCTIONS = {CONCATENATE: concatenate_loss, SEPARATE: separate_loss}

#: Values allowed for ``loss_reduction`` in the config.
LOSS_REDUCTIONS = tuple(LOSS_FUNCTIONS)


def build_loss_fn(reduction: str = CONCATENATE, line_lambda: list[float] | None = None):
    """Build the loss function from the config.

    The returned function is called as fn(logits, labels, token_weights, line_ids).
    Configuration errors are raised **here**, before the GPUs are allocated,
    not at the first step.
    """
    if reduction not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss_reduction: {reduction!r}. Valid values: {list(LOSS_REDUCTIONS)}."
        )
    if reduction == SEPARATE and not line_lambda:
        raise ValueError("The separate loss needs line_lambda (λ indexed by line ID).")
    return functools.partial(LOSS_FUNCTIONS[reduction], line_lambda=line_lambda)
