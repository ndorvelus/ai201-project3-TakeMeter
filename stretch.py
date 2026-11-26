"""TakeMeter stretch features.

    python stretch.py cv            # 5-fold CV -> out-of-fold predictions for all 150
    python stretch.py calibration   # confidence calibration (needs cv)
    python stretch.py patterns      # systematic error-pattern tests (needs cv)
    python stretch.py sheet         # blind 30-comment sheet for a second annotator
    python stretch.py agreement     # Cohen's kappa once the sheet is filled in

The headline test-set results come from takemeter.py; the cross-validation
predictions here are used only for the calibration and error-pattern analyses.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from takemeter import CSV_PATH, ID_TO_LABEL, LABEL_MAP, LABEL_NAMES, MODEL_NAME, RUNS

ROOT = Path(__file__).parent
OUT = ROOT / "results"
CV_PATH = OUT / "cv_predictions.json"
SHEET = ROOT / "second_annotator_sheet.csv"
SHEET_KEY = OUT / "second_annotator_key.json"


def load_all():
    df = pd.read_csv(CSV_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP).astype(int)
    return df


# ── Shared: 5-fold cross-validation ──────────────────────────────────────────
def cross_validate(k=5):
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)

    df = load_all()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    hp = RUNS["tuned"]

    def ds(d):
        return Dataset.from_pandas(
            d[["text", "label_id"]].rename(columns={"label_id": "labels"})
            .reset_index(drop=True)).map(
            lambda b: tok(b["text"], truncation=True, max_length=256),
            batched=True)

    records = [None] * len(df)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(df, df["label_id"])):
        set_seed(42)
        tr, val = train_test_split(df.iloc[tr_idx], test_size=0.15,
                                   random_state=42,
                                   stratify=df.iloc[tr_idx]["label_id"])
        ho = df.iloc[ho_idx]
        steps = int(np.ceil(len(tr) / 16)) * hp["num_train_epochs"]
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_MAP), id2label=ID_TO_LABEL,
            label2id=LABEL_MAP)
        args = TrainingArguments(
            output_dir=str(ROOT / "takemeter-model" / f"cv{fold}"),
            num_train_epochs=hp["num_train_epochs"],
            per_device_train_batch_size=16, per_device_eval_batch_size=32,
            learning_rate=hp["learning_rate"], weight_decay=0.01,
            warmup_steps=max(1, round(0.1 * steps)),
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="accuracy",
            report_to="none", seed=42, logging_steps=50)
        trainer = Trainer(
            model=model, args=args, train_dataset=ds(tr), eval_dataset=ds(val),
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=lambda p: {"accuracy": float(
                (p.predictions.argmax(-1) == p.label_ids).mean())})
        trainer.train()
        probs = torch.softmax(torch.tensor(
            trainer.predict(ds(ho)).predictions), -1).numpy()
        for row_pos, i in enumerate(ho_idx):
            p = probs[row_pos]
            records[i] = {
                "text": df.loc[i, "text"], "true": df.loc[i, "label"],
                "pred": ID_TO_LABEL[int(p.argmax())],
                "confidence": round(float(p.max()), 4), "fold": fold,
                "probs": {ID_TO_LABEL[j]: round(float(x), 4)
                          for j, x in enumerate(p)}}
        acc = np.mean([records[i]["pred"] == records[i]["true"]
                       for i in ho_idx])
        print(f"fold {fold}: held-out accuracy {acc:.3f} (n={len(ho_idx)})")

    OUT.mkdir(exist_ok=True)
    json.dump(records, open(CV_PATH, "w"), indent=2)
    overall = np.mean([r["pred"] == r["true"] for r in records])
    print(f"5-fold out-of-fold accuracy: {overall:.3f} on {len(records)}")


def load_cv():
    if not CV_PATH.exists():
        sys.exit("Run `python stretch.py cv` first.")
    return pd.DataFrame(json.load(open(CV_PATH)))


# ── Stretch 1: confidence calibration ────────────────────────────────────────
def calibration():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv = load_cv()
    cv["correct"] = cv["pred"] == cv["true"]
    edges = [0.0, 0.45, 0.55, 0.65, 1.0]
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = cv[(cv.confidence >= lo) & (cv.confidence < hi)]
        if len(b) == 0:
            continue
        conf, acc = b.confidence.mean(), b.correct.mean()
        ece += len(b) / len(cv) * abs(acc - conf)
        rows.append({"bin": f"{lo:.2f}–{hi:.2f}", "n": len(b),
                     "mean_confidence": round(conf, 3),
                     "accuracy": round(acc, 3),
                     "gap": round(acc - conf, 3)})
    corr = np.corrcoef(cv.confidence, cv.correct.astype(float))[0, 1]
    result = {"bins": rows, "ece": round(ece, 4),
              "mean_conf_correct": round(cv[cv.correct].confidence.mean(), 3),
              "mean_conf_wrong": round(cv[~cv.correct].confidence.mean(), 3),
              "conf_correct_correlation": round(float(corr), 3),
              "overall_accuracy": round(cv.correct.mean(), 3),
              "overall_mean_confidence": round(cv.confidence.mean(), 3)}
    json.dump(result, open(OUT / "calibration.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.3, 1], [0.3, 1], "--", color="gray", label="perfect calibration")
    ax.plot([r["mean_confidence"] for r in rows],
            [r["accuracy"] for r in rows], "o-", label="TakeMeter (5-fold CV)")
    for r in rows:
        ax.annotate(f"n={r['n']}", (r["mean_confidence"], r["accuracy"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0.3, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(ROOT / "calibration.png", dpi=150)
    print(json.dumps(result, indent=2))


# ── Stretch 2: error patterns ────────────────────────────────────────────────
STAT_RE = re.compile(
    r"\d|\b(ppg|apg|rpg|ts%|fg%|3pt|assists?|rebounds?|points?|stats?|"
    r"efficien\w*|percent|average\w*|record|wins?|games?|seasons?|mvp|"
    r"all[- ]?nba|all[- ]?star)\b", re.I)
REACT_RE = re.compile(
    r"\b(lol|lmao\w*|lmfao|smh|wtf|omg|bruh|holy|damn)\b|[!?]|"
    r"[\U0001F300-\U0001FAFF]|\b[A-Z]{4,}\b")


def rate(df):
    return {"n": int(len(df)),
            "accuracy": round(float((df.pred == df.true).mean()), 3)
            if len(df) else None}


def patterns():
    cv = load_cv()
    cv["has_stat"] = cv.text.apply(lambda t: bool(STAT_RE.search(t)))
    cv["has_react_marker"] = cv.text.apply(lambda t: bool(REACT_RE.search(t)))
    cv["short"] = cv.text.str.len() < 100

    ht = cv[cv.true == "hot_take"]
    h1 = {"hot_takes_with_stat_language": {
              **rate(ht[ht.has_stat]),
              "predicted_analysis": round(float(
                  (ht[ht.has_stat].pred == "analysis").mean()), 3)},
          "hot_takes_without": {
              **rate(ht[~ht.has_stat]),
              "predicted_analysis": round(float(
                  (ht[~ht.has_stat].pred == "analysis").mean()), 3)}}
    an = cv[cv.true == "analysis"]
    h1["analysis_without_stat_language"] = rate(an[~an.has_stat])
    h1["analysis_with_stat_language"] = rate(an[an.has_stat])

    rx = cv[cv.true == "reaction"]
    h2 = {"reactions_with_markers": rate(rx[rx.has_react_marker]),
          "reactions_without_markers": {
              **rate(rx[~rx.has_react_marker]),
              "predicted_hot_take": round(float(
                  (rx[~rx.has_react_marker].pred == "hot_take").mean()), 3)}}

    h3 = {"short_lt_100_chars": rate(cv[cv.short]),
          "long_ge_100_chars": rate(cv[~cv.short])}

    pairs = (cv[cv.pred != cv.true].groupby(["true", "pred"]).size()
             .sort_values(ascending=False))
    result = {"H1_stat_language": h1, "H2_reaction_markers": h2,
              "H3_length": h3,
              "error_pairs_true_to_pred": {f"{a}->{b}": int(n)
                                           for (a, b), n in pairs.items()},
              "total_errors": int((cv.pred != cv.true).sum())}
    json.dump(result, open(OUT / "error_patterns.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


# ── Stretch 4: second annotator ──────────────────────────────────────────────
def make_sheet():
    df = load_all()
    picked = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(10, random_state=7))
                .sample(frac=1, random_state=7).reset_index())
    sheet = pd.DataFrame({"id": range(1, len(picked) + 1),
                          "text": picked["text"], "label": ""})
    sheet.to_csv(SHEET, index=False)
    OUT.mkdir(exist_ok=True)
    json.dump({int(i): {"row": int(r), "my_label": l} for i, r, l in
               zip(sheet["id"], picked["index"], picked["label"])},
              open(SHEET_KEY, "w"), indent=2)
    print(f"Wrote {SHEET} ({len(sheet)} comments, labels hidden).")


def agreement():
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    sheet = pd.read_csv(SHEET).fillna("")
    key = json.load(open(SHEET_KEY))
    sheet["label"] = sheet["label"].str.strip().str.lower()
    done = sheet[sheet["label"] != ""]
    bad = set(done["label"]) - set(LABEL_MAP)
    if bad:
        sys.exit(f"Unknown labels in sheet: {bad}")
    if len(done) < 30:
        sys.exit(f"Only {len(done)}/30 comments labeled so far.")
    mine = [key[str(i)]["my_label"] for i in done["id"]]
    theirs = list(done["label"])
    kappa = cohen_kappa_score(mine, theirs)
    agree = np.mean([a == b for a, b in zip(mine, theirs)])
    result = {"n": len(done), "percent_agreement": round(float(agree), 3),
              "cohens_kappa": round(float(kappa), 3),
              "confusion_me_rows_them_cols": confusion_matrix(
                  mine, theirs, labels=LABEL_NAMES).tolist(),
              "disagreements": [{"text": t, "me": a, "them": b}
                                for t, a, b in zip(done["text"], mine, theirs)
                                if a != b]}
    json.dump(result, open(OUT / "agreement.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"},
                     indent=2))
    for d in result["disagreements"]:
        print(f"- me={d['me']:<9} them={d['them']:<9} {d['text'][:100]}")


if __name__ == "__main__":
    cmds = {"cv": cross_validate, "calibration": calibration,
            "patterns": patterns, "sheet": make_sheet, "agreement": agreement}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
