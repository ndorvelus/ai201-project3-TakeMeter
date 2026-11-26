"""TakeMeter web interface (stretch feature).

    .venv/bin/python app.py        # then open http://127.0.0.1:7860

Paste an r/nba comment; the fine-tuned model returns its label and the
confidence for all three labels. Needs the trained model from
`python takemeter.py finetune --run tuned`.
"""

from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "takemeter-model" / "tuned_best"
if not MODEL_DIR.exists():
    raise SystemExit("No trained model found. Run: python takemeter.py finetune --run tuned")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

DESCRIPTIONS = {
    "analysis": "backs a claim with specific, checkable evidence",
    "hot_take": "confident opinion without real support",
    "reaction": "in-the-moment emotional response",
}


def classify(text):
    text = (text or "").strip()
    if not text:
        return {}, ""
    with torch.no_grad():
        logits = model(**tokenizer(text, truncation=True, max_length=256,
                                   return_tensors="pt")).logits
    probs = torch.softmax(logits, -1)[0].tolist()
    scores = {model.config.id2label[i]: p for i, p in enumerate(probs)}
    top = max(scores, key=scores.get)
    note = (f"**{top}** ({scores[top]:.0%}): {DESCRIPTIONS[top]}.\n\n"
            "Confidence scores from this model are poorly calibrated "
            "(see README), so treat them as a rough ranking, not a probability.")
    return scores, note


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="r/nba comment"),
    outputs=[gr.Label(num_top_classes=3, label="Prediction"),
             gr.Markdown()],
    title="TakeMeter: is this r/nba comment analysis, a hot take, or a reaction?",
    description="Fine-tuned DistilBERT trained on 105 labeled r/nba comments. "
                "Test accuracy 0.57; a zero-shot LLM baseline scored 0.74.",
    examples=[
        ["Jokic is averaging 29/13/10 on 63% true shooting, there's no MVP debate"],
        ["LeBron is washed, he can't guard anyone anymore"],
        ["LMAOOO did you see that dunk?? I'm screaming"],
        ["Bane is averaging 19 a game and making over 4 3's a game on pretty elite efficiency."],
        ["Derrick White lowkey been ass since the beginning of last year."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
