"""TakeMeter pipeline — a local mirror of the AI201 starter notebook.

Same split (70/15/15, stratified, random_state=42), same model
(distilbert-base-uncased), same Trainer setup and metrics as the Colab
notebook, so results are comparable. Runs on Apple Silicon (MPS), CUDA, or CPU.

Usage:
    python takemeter.py finetune --run default     # notebook defaults
    python takemeter.py finetune --run tuned       # fixed warmup + more epochs
    python takemeter.py baseline                   # needs GROQ_API_KEY in .env
    python takemeter.py report                     # writes evaluation_results.json
    python takemeter.py classify "some comment"    # demo: label + confidence
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "takemeter_nba_labeled.csv"
OUT = ROOT / "results"
MODEL_NAME = "distilbert-base-uncased"

LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]

# Hyperparameter configurations. "default" is copied from the notebook.
RUNS = {
    "default": dict(num_train_epochs=3, learning_rate=2e-5, warmup_steps=50),
    # With 105 training examples and batch size 16 there are only 7 steps per
    # epoch (21 total for 3 epochs), so warmup_steps=50 means the learning rate
    # never reaches its target. "tuned" warms up over 10% of steps instead and
    # trains longer; best epoch is chosen on the validation set.
    # 7 warmup steps = 10% of 70 total steps (7 steps/epoch x 10 epochs).
    "tuned": dict(num_train_epochs=10, learning_rate=5e-5, warmup_steps=7),
}


def load_splits():
    df = pd.read_csv(CSV_PATH)
    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        sys.exit(f"Labels in CSV not in LABEL_MAP: {unknown}")
    df["label_id"] = df["label"].map(LABEL_MAP).astype(int)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def metrics_block(true_ids, pred_ids):
    report = classification_report(
        true_ids, pred_ids, labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES, zero_division=0, output_dict=True)
    return {
        "accuracy": round(accuracy_score(true_ids, pred_ids), 4),
        "macro_f1": round(f1_score(true_ids, pred_ids, average="macro",
                                   zero_division=0), 4),
        "per_class": {k: {m: round(v[m], 4) for m in
                          ("precision", "recall", "f1-score", "support")}
                      for k, v in report.items() if k in LABEL_NAMES},
        "confusion_matrix": confusion_matrix(
            true_ids, pred_ids, labels=list(range(len(LABEL_NAMES)))).tolist(),
    }


# ── Fine-tuning ──────────────────────────────────────────────────────────────
def finetune(run_name):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)

    set_seed(42)
    train_df, val_df, test_df = load_splits()
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def make_dataset(d):
        ds = Dataset.from_pandas(
            d[["text", "label_id"]].rename(columns={"label_id": "labels"}))
        return ds.map(lambda b: tokenizer(b["text"], truncation=True,
                                          max_length=256), batched=True)

    train_ds, val_ds, test_ds = map(make_dataset, (train_df, val_df, test_df))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP), id2label=ID_TO_LABEL,
        label2id=LABEL_MAP)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=-1))}

    hp = RUNS[run_name]
    args = TrainingArguments(
        output_dir=str(ROOT / "takemeter-model" / run_name),
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=hp["learning_rate"],
        weight_decay=0.01,
        warmup_steps=hp["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=5,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()

    val_acc = trainer.evaluate(val_ds)["eval_accuracy"]
    out = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
    pred_ids = probs.argmax(-1)
    true_ids = out.label_ids

    OUT.mkdir(exist_ok=True)
    result = {"run": run_name, "hyperparameters": hp,
              "val_accuracy": round(float(val_acc), 4),
              "test": metrics_block(true_ids, pred_ids)}
    preds = [{"text": test_df.loc[i, "text"],
              "true": ID_TO_LABEL[int(true_ids[i])],
              "pred": ID_TO_LABEL[int(pred_ids[i])],
              "confidence": round(float(probs[i][pred_ids[i]]), 4),
              "probs": {ID_TO_LABEL[j]: round(float(p), 4)
                        for j, p in enumerate(probs[i])}}
             for i in range(len(test_df))]
    json.dump(result, open(OUT / f"finetune_{run_name}.json", "w"), indent=2)
    json.dump(preds, open(OUT / f"finetune_{run_name}_predictions.json", "w"),
              indent=2)
    trainer.save_model(str(ROOT / "takemeter-model" / f"{run_name}_best"))
    tokenizer.save_pretrained(str(ROOT / "takemeter-model" / f"{run_name}_best"))

    print(f"\n[{run_name}] val acc {val_acc:.3f} | test acc "
          f"{result['test']['accuracy']:.3f} | macro-F1 "
          f"{result['test']['macro_f1']:.3f}")
    print(classification_report(true_ids, pred_ids, target_names=LABEL_NAMES,
                                zero_division=0))


# ── Groq baseline ────────────────────────────────────────────────────────────
# Prompt examples are all from the TRAIN split, so no test comment leaks in.
SYSTEM_PROMPT = """You are classifying comments from r/nba, the NBA subreddit.
Assign each comment to exactly one of the following categories.

analysis: The comment makes a claim and backs it with specific, checkable evidence (stats, records, history, contracts, or concrete tactical/physical observations) that would still support the point if the opinion framing were removed.
Example: "Bane is averaging 19 a game and making over 4 3's a game on pretty elite efficiency. The contract isn't great but it's also not terrible for that kinda production."

hot_take: The comment confidently asserts an opinion, judgment, or prediction without real supporting evidence, or with evidence that is vague or a single fact picked for effect rather than reasoned with.
Example: "Derrick White lowkey been ass since the beginning of last year."

reaction: The comment is an immediate emotional response (excitement, anger, amusement, sarcasm) to a specific game, play, clip, or piece of news, with little or no argument.
Example: "Bout time these 2 teams played comp. Hope they enjoyed their 3 month preseason"

Rules for borderline comments:
- A stat used to vent or marvel about something that just happened is reaction.
- A bold claim resting on one cherry-picked fact is hot_take.

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
"""

BASELINE_MODELS = {
    # Used in the starter notebook's Section 5. The project page names
    # meta-llama/llama-4-scout-17b-16e-instruct, but Groq now returns 404
    # model_not_found for it, so the notebook's model is the baseline.
    "gpt_oss_120b": "openai/gpt-oss-120b",
}


def baseline():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY not set. Put it in .env (GROQ_API_KEY=...).")
    client = Groq(api_key=key, timeout=60)
    _, _, test_df = load_splits()
    OUT.mkdir(exist_ok=True)

    for short, model in BASELINE_MODELS.items():
        print(f"\nBaseline {model} on {len(test_df)} test comments...")
        preds = []
        for text in test_df["text"]:
            label = None
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=model, temperature=0,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",
                                   "content": f"Classify this post:\n\n{text}"}])
                    raw = (r.choices[0].message.content or "").strip().lower()
                    for lab in sorted(LABEL_MAP, key=len, reverse=True):
                        if raw == lab or lab in raw:
                            label = lab
                            break
                    break
                except Exception as e:  # rate limits on the free tier
                    if getattr(e, "status_code", None) in (400, 401, 403, 404):
                        sys.exit(f"Groq error for {model}: {e}")
                    print(f"  API error ({e}); retrying")
                    time.sleep(5 * (attempt + 1))
            preds.append(label)
            time.sleep(0.3)

        unparsed = preds.count(None)
        valid = [(LABEL_MAP[p], t) for p, t in zip(preds, test_df["label_id"])
                 if p is not None]
        result = {"model": model, "unparseable": unparsed,
                  "evaluated": len(valid),
                  "test": metrics_block([t for _, t in valid],
                                        [p for p, _ in valid])}
        json.dump(result, open(OUT / f"baseline_{short}.json", "w"), indent=2)
        json.dump([{"text": t, "true": l, "pred": p} for t, l, p in
                   zip(test_df["text"], test_df["label"], preds)],
                  open(OUT / f"baseline_{short}_predictions.json", "w"),
                  indent=2)
        print(f"  accuracy {result['test']['accuracy']:.3f} | macro-F1 "
              f"{result['test']['macro_f1']:.3f} | unparseable {unparsed}")


# ── Report + confusion matrix ────────────────────────────────────────────────
def report(final_run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ft = json.load(open(OUT / f"finetune_{final_run}.json"))
    cm = np.array(ft["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png", dpi=150)

    baselines = {}
    for short in BASELINE_MODELS:
        p = OUT / f"baseline_{short}.json"
        if p.exists():
            baselines[short] = json.load(open(p))
    primary = baselines.get("gpt_oss_120b")
    _, _, test_df = load_splits()
    results = {
        "baseline_accuracy": primary and primary["test"]["accuracy"],
        "finetuned_accuracy": ft["test"]["accuracy"],
        "improvement": primary and round(
            ft["test"]["accuracy"] - primary["test"]["accuracy"], 4),
        "test_set_size": len(test_df),
        "label_map": LABEL_MAP,
        "model": MODEL_NAME,
        "finetune_run": final_run,
        "finetune_hyperparameters": ft["hyperparameters"],
        "finetuned_detail": ft["test"],
        "baselines": {k: {"model": v["model"], "unparseable": v["unparseable"],
                          **v["test"]} for k, v in baselines.items()},
    }
    json.dump(results, open(ROOT / "evaluation_results.json", "w"), indent=2)
    print("Wrote evaluation_results.json and confusion_matrix.png")


# ── Demo ─────────────────────────────────────────────────────────────────────
def classify(texts, model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    for text in texts:
        with torch.no_grad():
            logits = model(**tok(text, truncation=True, max_length=256,
                                 return_tensors="pt")).logits
        probs = torch.softmax(logits, -1)[0]
        i = int(probs.argmax())
        print(f"\n{text}\n → {model.config.id2label[i]} "
              f"({probs[i]:.0%} confident)  | " + "  ".join(
                  f"{model.config.id2label[j]} {p:.0%}"
                  for j, p in enumerate(probs.tolist())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finetune"); f.add_argument("--run", choices=RUNS,
                                                   default="default")
    sub.add_parser("baseline")
    r = sub.add_parser("report"); r.add_argument("--run", default="tuned")
    c = sub.add_parser("classify"); c.add_argument("texts", nargs="+")
    c.add_argument("--model", default=str(ROOT / "takemeter-model" /
                                          "tuned_best"))
    a = ap.parse_args()
    if a.cmd == "finetune":
        finetune(a.run)
    elif a.cmd == "baseline":
        baseline()
    elif a.cmd == "report":
        report(a.run)
    else:
        classify(a.texts, a.model)
