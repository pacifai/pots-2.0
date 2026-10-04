"""The per-class residual table and the per-role ρ table (A10, reused by A11)."""

import math

from verification.verifier.context import ProductStat, StepStats, TensorStat
from verification.verifier.residuals import (
    class_summary,
    format_class_table,
    format_tensor_table,
    tensor_summary,
    weight_role,
)


def stats():
    s1 = StepStats(
        products=[ProductStat(1, "L1.Y_q", "Y_q", 2.0, (1.0, 1.0)),
                  ProductStat(2, "L1.S[0,0]", "S", 4.0, (3.0, 0.0)),
                  ProductStat(3, "L2.Y_q", "Y_q", 6.0, (0.0, 2.0))],
        tensors=[TensorStat("model.layers.0.self_attn.q_proj.weight", "6a", 1.5),
                 TensorStat("model.layers.1.self_attn.q_proj.weight", "6a", 1.9),
                 TensorStat("model.norm.weight", "6b", 0.0)])
    s2 = StepStats(
        products=[ProductStat(1, "L1.Y_q", "Y_q", 10.0, (4.0, 0.0))],
        tensors=[TensorStat("model.layers.0.self_attn.q_proj.weight", "6a", 1.95)])
    return {2: s2, 1: s1}


def test_weight_role_stars_the_layer_index():
    assert weight_role("model.layers.13.mlp.up_proj.weight") == "model.layers.*.mlp.up_proj.weight"
    assert weight_role("model.embed_tokens.weight") == "model.embed_tokens.weight"


def test_class_summary_pools_every_residual_of_every_step():
    rows = {r.cls: r for r in class_summary(stats())}
    assert list(rows) == ["Y_q", "S"]  # first appearance, steps in order
    q = rows["Y_q"]
    assert (q.count, q.n) == (3, 6)
    assert math.isclose(q.rms, math.sqrt((1 + 1 + 0 + 4 + 16 + 0) / 6))
    assert (q.max, q.max_name, q.max_step) == (4.0, "L1.Y_q", 2)
    assert (q.kappa_median, q.kappa_max) == (6.0, 10.0)
    s = rows["S"]
    assert (s.count, s.n, s.max, s.max_step) == (1, 2, 3.0, 1)
    assert math.isclose(s.rms, math.sqrt(9 / 2))
    # an iterable of (step, stats) gives the same rows
    assert class_summary(sorted(stats().items())) == class_summary(stats())


def test_a_non_finite_residual_shows_in_its_class():
    st = StepStats(products=[ProductStat(1, "a", "c", 1.0, (1.0, math.nan, 2.0))])
    (r,) = class_summary({1: st})
    assert math.isnan(r.rms) and math.isnan(r.max)
    st = StepStats(products=[ProductStat(1, "a", "c", 3.0, (1.0,)),
                             ProductStat(2, "b", "c", math.nan, (1.0,))])
    (r,) = class_summary({1: st})
    assert math.isnan(r.kappa_median) and math.isnan(r.kappa_max) and r.rms == 1.0


def test_tensor_summary_groups_by_check_and_role():
    rows = tensor_summary(stats())
    assert [(r.check_id, r.role, r.count) for r in rows] == [
        ("6a", "model.layers.*.self_attn.q_proj.weight", 3), ("6b", "model.norm.weight", 1)]
    assert (rows[0].rho_max, rows[0].max_name, rows[0].max_step) == (
        1.95, "model.layers.0.self_attn.q_proj.weight", 2)


def test_tables_print_one_line_per_row_and_the_summaries():
    lines = []
    format_class_table(class_summary(stats()), lines.append)
    assert len(lines) == 1 + 2 + 1
    assert lines[-1].startswith("largest class RMS 2.121 (S); global max 4.000 (L1.Y_q)")
    lines = []
    format_tensor_table(tensor_summary(stats()), lines.append)
    assert len(lines) == 1 + 2 + 2
    assert lines[-2:] == ["check 6a: max ρ 1.950 (model.layers.0.self_attn.q_proj.weight)",
                          "check 6b: max ρ 0.000 (model.norm.weight)"]
