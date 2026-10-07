#!/usr/bin/env python3
"""Per-item (b) vs (c) breakdown for the REPORT's qualitative section.

**Why this exists.** NB5 §5 writes `results/qualitative.json` with the fine-tune's
per-item `target` score but *not* baseline (b)'s. The report template asks for a row per
ticket with both "(b) prompt" and "(c) fine-tune" answers, and the rubric requires ≥2
cases where the fine-tune LOSES. Without (b)'s per-item answer, "the fine-tune lost here"
cannot be asserted from the artifacts — only guessed from a low absolute score, which is
exactly the kind of unfounded claim this lab is about.

So this script re-decodes **both** arms on **both** graded groups, per item:

    target      (50 tickets)  (b) base+OPTIMIZED_PROMPT      vs  (c) base+adapters/correct
                              scored with triage_field_accuracy
    regression  (15 items)    (b) base, system=None          vs  (c) base+adapters/correct
                              scored with keyword_recall

`(c)` is always scored the way NB5 scores it: `NAIVE_PROMPT` as the system message.

Greedy decoding is deterministic, so re-decoding must reproduce the frozen numbers; that
is asserted, not assumed. Nothing here modifies the frozen baseline — it re-measures it
and adds the per-item detail NB5 does not keep.

**Why regression is in here.** On the target group the fine-tune wins 48 of 50 items and
loses none, so the rubric's "≥2 fine-tune losses" cannot be sourced there. The losses it
*does* have are on the regression group — 14 of 15 items — and they are the entire reason
NB5's verdict is FAILED. Sourcing the loss cases from the group where the loss actually
happened is the point; picking the three lowest target scores instead and calling them
losses would be a fabrication.

Writes (superset of NB5's schema, so `qualitative.json` stays backwards compatible):

    results/qualitative.json

    python scripts/qualitative_breakdown.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report  # noqa: E402
from labkit.config import get_tier                   # noqa: E402

OUTCOMES = ("ft_win", "tie", "ft_loss")


def load_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def outcome(s_ft: float, s_b: float) -> str:
    return "ft_win" if s_ft > s_b else ("ft_loss" if s_ft < s_b else "tie")


def counts(rows: list[dict]) -> dict:
    return {k: sum(1 for x in rows if x["outcome"] == k) for k in OUTCOMES}


def main() -> int:
    from peft import PeftModel

    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
    frozen = json.loads((ROOT / "results" / "baselines_frozen.json").read_text(encoding="utf-8"))
    want_b = frozen["baseline_b"]["target"]
    want_b_reg = frozen["baseline_b"]["regression"]
    print(f"model={tier.model_id}  target={len(target)}  regression={len(regression)}")
    print(f"frozen (b): target={want_b:.4f}  regression={want_b_reg:.4f}")

    t_prompts = [r["input"] for r in target]
    r_prompts = [r["instruction"] for r in regression]

    # --- (b) base + optimized prompt (target) / plain base (regression) --------
    model, tok = generate.load_base(tier)
    preds_b, _ = generate.generate_batch(model, tok, t_prompts,
                                         system=generate.OPTIMIZED_PROMPT,
                                         label="(b) base+optimized/target")
    rpreds_b, _ = generate.generate_batch(model, tok, r_prompts, system=None,
                                          max_new_tokens=96, label="(b)/regression")
    del model
    generate.free_memory()

    # --- (c) base + adapters/correct, NAIVE_PROMPT (NB5's evaluation contract) --
    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    preds_ft, _ = generate.generate_batch(model, tok, t_prompts,
                                          system=generate.NAIVE_PROMPT,
                                          label="(c) LoRA fine-tune/target")
    rpreds_ft, _ = generate.generate_batch(model, tok, r_prompts, system=None,
                                           max_new_tokens=96, label="(c)/regression")
    del model
    generate.free_memory()

    # --- target rows ----------------------------------------------------------
    rows = []
    for i, (r, pb, pft) in enumerate(zip(target, preds_b, preds_ft)):
        sb = ev.triage_field_accuracy(pb, r["label"])
        sft = ev.triage_field_accuracy(pft, r["label"])
        # per-field detail, so the report can say WHICH field went wrong.
        # keys=[k] is load-bearing: with the default key list the other three fields have
        # no reference and are skipped, so a correct field would score 1/4, not 1.0.
        fb = {k: ev.triage_field_accuracy(pb, r["label"], keys=[k]) for k in ev.TRIAGE_KEYS}
        fft = {k: ev.triage_field_accuracy(pft, r["label"], keys=[k]) for k in ev.TRIAGE_KEYS}
        rows.append({
            "i": i,
            "ticket": r["input"],
            "label": r["label"],
            "b_pred": " ".join(pb.split()),
            "ft_pred": " ".join(pft.split()),
            "b_score": round(sb, 2),
            "ft_score": round(sft, 2),
            "delta": round(sft - sb, 2),
            "b_fields": {k: round(v, 2) for k, v in fb.items()},
            "ft_fields": {k: round(v, 2) for k, v in fft.items()},
            "outcome": outcome(sft, sb),
        })

    # --- regression rows ------------------------------------------------------
    rrows = []
    for i, (r, pb, pft) in enumerate(zip(regression, rpreds_b, rpreds_ft)):
        sb = ev.keyword_recall(pb, r["keywords"])
        sft = ev.keyword_recall(pft, r["keywords"])
        rrows.append({
            "i": i,
            "question": r["instruction"],
            "keywords": r["keywords"],
            "b_pred": " ".join(pb.split())[:300],
            "ft_pred": " ".join(pft.split())[:300],
            "b_score": round(sb, 2),
            "ft_score": round(sft, 2),
            "delta": round(sft - sb, 2),
            "outcome": outcome(sft, sb),
        })

    n, nr = len(rows), len(rrows)
    mean_b = sum(x["b_score"] for x in rows) / n
    mean_ft = sum(x["ft_score"] for x in rows) / n
    mean_b_reg = sum(x["b_score"] for x in rrows) / nr
    mean_ft_reg = sum(x["ft_score"] for x in rrows) / nr
    c_t, c_r = counts(rows), counts(rrows)

    print(f"\nre-decoded (b) target      = {mean_b:.4f}  (frozen {want_b:.4f})")
    print(f"re-decoded (c) target      = {mean_ft:.4f}")
    print(f"target outcomes:      {c_t}")
    print(f"re-decoded (b) regression  = {mean_b_reg:.4f}  (frozen {want_b_reg:.4f})")
    print(f"re-decoded (c) regression  = {mean_ft_reg:.4f}")
    print(f"regression outcomes:  {c_r}")

    # The frozen bar must reproduce, or this breakdown is measuring something else.
    for label, got, want in (("target", mean_b, want_b), ("regression", mean_b_reg, want_b_reg)):
        if abs(got - want) > 0.01:
            print(f"\n!! (b) {label} did not reproduce the frozen value "
                  f"({got:.4f} vs {want:.4f}). Do NOT quote this table until it is explained.")
            return 1

    # Wins/losses are selected by the measured delta, extremes first, ties broken by
    # item index — a deterministic rule, so the report's cases are not hand-picked.
    def extremes(items, kind):
        sel = [x for x in items if x["outcome"] == kind]
        key = (lambda x: (-x["delta"], x["i"])) if kind == "ft_win" else (lambda x: (x["delta"], x["i"]))
        return sorted(sel, key=key)

    payload = {
        "method": "greedy re-decode of arm (b) and arm (c) on BOTH graded groups; "
                  "target score = triage_field_accuracy, regression score = keyword_recall; "
                  "(c) always uses NAIVE_PROMPT, as NB5 scores it",
        "model": tier.model_id,
        "target": {
            "n": n,
            "reproduced_b_target": round(mean_b, 4),
            "frozen_b_target": round(want_b, 4),
            "reproduced_ft_target": round(mean_ft, 4),
            "counts": c_t,
            "rows": rows,
            "win_examples": extremes(rows, "ft_win")[:6],
            "loss_examples": extremes(rows, "ft_loss")[:6],
        },
        "regression": {
            "n": nr,
            "reproduced_b_regression": round(mean_b_reg, 4),
            "frozen_b_regression": round(want_b_reg, 4),
            "reproduced_ft_regression": round(mean_ft_reg, 4),
            "counts": c_r,
            "rows": rrows,
            "win_examples": extremes(rrows, "ft_win")[:6],
            "loss_examples": extremes(rrows, "ft_loss")[:6],
        },
        "selection_rule": "sorted by per-item delta (ft_score - b_score); wins descending, "
                          "losses ascending; ties broken by item index",
        # NB5 wrote a flat array of target rows. Keep that access path working so nothing
        # reading `qualitative.json` as a list of per-ticket rows breaks.
        "rows": rows,
    }
    report.write_json(payload, "qualitative.json", results_dir=ROOT / "results")
    print(f"\nwrote results/qualitative.json  (target {c_t}; regression {c_r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
