# TakeMeter: r/nba Take Classifier

TakeMeter is a fine-tuned DistilBERT classifier that sorts r/nba comments into **analysis**, **hot_take**, or **reaction**, compared against a zero-shot Groq LLM baseline on the same held-out test set.

- **Dataset:** [`takemeter_nba_labeled.csv`](takemeter_nba_labeled.csv) (150 labeled comments)
- **Planning doc:** [`planning.md`](planning.md)
- **Pipeline:** [`takemeter.py`](takemeter.py) (a local mirror of the starter notebook, see [Running it](#running-it))
- **Outputs:** [`evaluation_results.json`](evaluation_results.json), [`confusion_matrix.png`](confusion_matrix.png), per-example predictions in [`results/`](results/)

---

## Community

**r/nba** is one of the largest sports communities on Reddit. Every game, trade, and injury produces thousands of comments, from stat-backed breakdowns to one-line trash talk. Regulars already police the difference between these modes ("source?", "that's a hot take", "this is just a reaction"), so the labels measure a distinction the community actually cares about: does a comment *argue*, *assert*, or *vent*?

## Label taxonomy

| Label | Definition | Examples |
|---|---|---|
| **analysis** | Makes a claim and backs it with specific, checkable evidence (stats, records, history, contracts, or concrete tactical/physical observations) that would still support the point with the opinion framing removed. | "Tbf his efficiency is improving in recent games because he shooting more 3s and less midrange jumpers… the Celtics tend to do better when he's taking at least 10 3s a game historically." · "Bane is averaging 19 a game and making over 4 3's a game on pretty elite efficiency. The contract isn't great but it's also not terrible for that kinda production." |
| **hot_take** | Confidently asserts an opinion, judgment, or prediction without real supporting evidence, or with evidence that is vague or a single fact picked for effect. | "Bones isn't a lazy defender, he's just bad. Effort gets you pretty far in the regular season." · "Danny Green is terrible. Farmar was better, not sure why he only got one shot (so far)" |
| **reaction** | An immediate emotional response (excitement, anger, amusement, sarcasm) to a specific game, play, clip, or piece of news, with little or no argument. | "I wish I could upvote this post twice. Shamgod deserves all the respect and flowers in the world" · "Bout time these 2 teams played comp. Hope they enjoyed their 3 month preseason" |

**Decision rules for boundaries:** one cherry-picked stat under a bold claim → hot_take; a stat used to vent or marvel about something that just happened → reaction; specific eye-test observations (checkable on film) → analysis; sarcasm aimed at a specific event → reaction. Full reasoning is in [`planning.md`](planning.md).

## Data

**Source.** Public r/nba comments. Reddit blocks anonymous automated access (403s, or a redirect to login), so comments were pulled from the [Arctic Shift](https://arctic-shift.photon-reddit.com) public Reddit archive. Every row keeps a `source_url` linking to the original comment; usernames were not stored.

**Collection.** About 1,000 comments sampled across 10 different days over two weeks, so no single game or storyline dominates. Filtered out bots (AutoModerator), deleted/removed comments, comments with links, and comments under 60 or over 600 characters, leaving 613 candidates. Most candidates were *not* basketball takes (politics, personal insults, arguments about the subreddit), so I kept only comments about basketball. The 90%-coverage rule applies to basketball comments, not every comment on the sub.

**Labeling process.** Claude pre-labeled each selected comment using the definitions above and flagged 21 borderline cases with a `HARD:` note explaining the rule applied.

**Label distribution (150 total):**

| Label | Count | Share |
|---|---|---|
| analysis | 51 | 34% |
| hot_take | 51 | 34% |
| reaction | 48 | 32% |

The notebook split (stratified, `random_state=42`) gives **105 train / 22 validation / 23 test**.

> ⚠️ The project asks for at least 200 examples; this dataset has 150. The main cost is a 23-comment test set, where one comment ≈ 4.3 accuracy points, so differences of one or two comments between models are noise.

**Three difficult-to-label examples:**

1. **"Number of NBA level players: a lot. Number of people who won the dunk contest 3 times in a row: 1. What McClung did is way more impressive."** This could be analysis (it cites a fact) or hot_take. The single stat is chosen to win the argument rather than to reason with, so it went to **hot_take**, the same rule as the LeBron example on the project page.
2. **"Grizz are like 2nd or third in 3pt% and 1st in FT% and I honestly have never seen numbers like that in this franchise's history."** This has real, checkable stats, but the point of the comment is disbelief, not an argument. Labeled **reaction**.
3. **"Not trying to be a downer, but I had this injury and surgery at 24, and it was a year recovery… There's a chance this may be his career tbh."** This is specific and detailed, but the evidence is a personal anecdote that can't be checked against this player. Labeled **hot_take**.

## Fine-tuning

- **Base model:** `distilbert-base-uncased` with a 3-way classification head.
- **Setup:** Hugging Face `Trainer`, same split, tokenization (max 256 tokens), batch size 16, weight decay 0.01, per-epoch evaluation, and best-checkpoint-by-validation-accuracy as the starter notebook. Run locally on an Apple Silicon GPU (MPS) via [`takemeter.py`](takemeter.py) instead of Colab.

**Key hyperparameter decision: warmup.** The notebook default `warmup_steps=50` is longer than the entire training run. 105 examples at batch size 16 is 7 steps per epoch, so only 21 steps across 3 epochs, which means the learning rate never finishes warming up to 2e-5. The default run's final training loss was **1.087**, essentially ln(3) ≈ 1.099, the loss of guessing uniformly among three labels. I changed to:

| Setting | Notebook default | Final (tuned) | Why |
|---|---|---|---|
| warmup | 50 steps | 7 steps (10% of 70) | Let the LR actually reach its target |
| epochs | 3 | 10 | 21 steps is too few updates for a new classifier head |
| learning rate | 2e-5 | 5e-5 | Small dataset, few steps; the upper end of the usual BERT range |

The tuned configuration was chosen on **validation** accuracy (0.682 → 0.773) before looking at its test results.

## Baseline

Zero-shot classification with Groq, `temperature=0`, one call per test comment. The prompt contains the three label definitions, one example per label, the two boundary rules, and an instruction to output only the label name. **All prompt examples are taken from the training split.** My first draft used the Mo Wagner comment, which turned out to be in the test set, so I swapped it out to avoid handing the baseline a test answer. The full prompt is `SYSTEM_PROMPT` in [`takemeter.py`](takemeter.py).

**Model:** `openai/gpt-oss-120b`, the model the starter notebook's Section 5 calls. The project page names `meta-llama/llama-4-scout-17b-16e-instruct`, but Groq now returns `404 model_not_found` for it, so it couldn't be run. All 23 responses parsed cleanly to a label. Per-comment baseline predictions are in [`results/baseline_gpt_oss_120b_predictions.json`](results/baseline_gpt_oss_120b_predictions.json).

## Evaluation report

### Overall

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| **Zero-shot gpt-oss-120b (Groq)** | **0.739** | **0.735** | 17 / 23 correct, 0 unparseable |
| DistilBERT, notebook defaults | 0.522 | 0.491 | Barely trained (loss ≈ chance) |
| DistilBERT, tuned (final) | 0.565 | 0.571 | 13 / 23 correct |

**Fine-tuning did not beat the baseline.** The zero-shot LLM is ahead by 4 comments (17.4 accuracy points). With a 23-comment test set that's not a precise gap, but it is consistent across every class, so I read it as real rather than noise.

### Per-class (test set, n = 23)

| Label | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Baseline P | Baseline R | Baseline F1 |
|---|---|---|---|---|---|---|
| analysis (n=7) | 0.57 | 0.57 | 0.57 | 0.80 | 0.57 | 0.67 |
| hot_take (n=8) | 0.45 | 0.62 | 0.53 | 0.64 | 0.88 | 0.74 |
| reaction (n=8) | 0.80 | 0.50 | 0.62 | 0.86 | 0.75 | 0.80 |

Baseline confusion matrix (rows = true; columns = analysis / hot_take / reaction): analysis 4 / 3 / 0, hot_take 0 / 7 / 1, reaction 1 / 1 / 6. Both models struggle most with analysis recall (4 of 7 each), and both push missed analysis comments into hot_take.

### Confusion matrix: fine-tuned model (rows = true, columns = predicted)

| true ↓ / predicted → | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | **4** | 3 | 0 |
| **hot_take** | 2 | **5** | 1 |
| **reaction** | 1 | 3 | **4** |

hot_take is the model's default guess when unsure: it absorbs 6 of the 10 errors (3 from analysis, 3 from reaction), which is why hot_take has the highest recall but the lowest precision. The model never confused analysis with reaction in the analysis→reaction direction. Image copy: [`confusion_matrix.png`](confusion_matrix.png).

### Wrong predictions, analyzed

**1. "This is what small sample sizes look like. Mo Wagner is currently shooting 64% from 3 this year."** True **analysis** → predicted **hot_take** (0.37, the lowest-confidence prediction on the test set).
The confusion is on the analysis/hot_take boundary, and it's partly my own label design. This is a short post with one stat, which is structurally identical to the McClung dunk-contest comment I labeled hot_take. The difference is *what the stat does*: here the stat *is* the argument (64% from 3 is an obvious small-sample outlier), while the McClung stat decorates a claim. That distinction depends on basketball knowledge DistilBERT doesn't have, and on reading intent from one sentence. **Fix:** more short, single-stat analysis examples, or a tighter rule (e.g., "one stat is analysis when the stat directly demonstrates the claim").

**2. "Yao would definitely have more assists and ppg (some from 3s) if he played today instead of two decades ago."** True **hot_take** → predicted **analysis** (0.65, one of its *most* confident predictions).
The comment is full of stat *vocabulary* (assists, ppg, 3s, "two decades ago") but contains no actual evidence; it's a confident counterfactual. The model learned "stat words → analysis" rather than "evidence → analysis." The Bosh/Mobley comparison (#20 in the test predictions, also hot_take → analysis at 0.64) fails the same way: comparison language reads like reasoning. This is a data problem, not an annotation inconsistency. Those were labeled consistently, but the training set has few hot_takes that *sound* analytical. **Fix:** deliberately collect stat-flavored hot_takes.

**3. "I wish that there would be a single game, just one where he would shut his yap and would just play without constant whining."** True **reaction** → predicted **hot_take** (0.41).
This one is closer to an annotation problem. My reaction definition requires a response to *a specific* game or event, and this comment is about a player's behavior in general, which makes it arguably a hot_take by my own rules. Reaction recall was only 0.50 overall: the reactions the model caught mostly have strong surface markers ("lol", "lmao", "Alright Reddit, do your thing"), while calmer ones like "Yeah, I'm really looking forward to seeing more Paolo in the new offense…" (also → hot_take) were missed. **Fix:** tighten the reaction definition around "is there a specific triggering event," and re-review reaction labels for consistency.

### Sample classifications (fine-tuned model)

| Post (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| Tbf his efficiency is improving in recent games because he shooting more 3s and less midrange… | analysis | analysis | 0.73 | yes |
| The west had a better record against the east last year for the 24th time in 27 years… | analysis | analysis | 0.64 | yes |
| Bam & Kobe in the same headline. Alright Reddit, do your thing. | reaction | reaction | 0.58 | yes |
| Yao would definitely have more assists and ppg (some from 3s) if he played today… | hot_take | analysis | 0.65 | no |
| One of the worst I've ever seen. Ball never touched the court after about the half way line. | reaction | analysis | 0.44 | no |

**Why the first prediction is reasonable:** the comment states a cause ("shooting more 3s and less midrange"), backs it with a player-specific tendency, and cites a team-level pattern ("at least 10 3s a game historically"). It's exactly the claim-plus-checkable-evidence structure the analysis label describes, and it's the model's highest-confidence prediction on the test set.

**Confidence:** predictions are uniformly low-confidence (0.37–0.73). Correct predictions averaged 0.55 and wrong ones 0.47, so confidence carries a little signal, but not enough to trust any single prediction. Wrong predictions were also shorter (median 116 characters vs. 168 for correct ones), which fits failures 1 and 3: short posts give the model little to go on.

### What the model learned vs. what I intended

I intended the model to learn *how a comment supports its claim*: evidence vs. assertion vs. emotion. What it mostly learned is *surface style*. Numbers and stat vocabulary push toward analysis even when nothing is being proven (Yao, Bosh). Internet-reaction markers (lol, lmao, caps, "Alright Reddit") push toward reaction, so calm reactions get missed. hot_take became the fallback for anything that matched neither style. That's a sensible strategy for 105 training examples, since style is a cheap proxy that correlates with my labels, but it's not the distinction I care about. The analysis/hot_take boundary in particular depends on judging whether evidence is *relevant and sufficient*, which needs basketball knowledge a small model can't pick up from 35 examples per class.

The baseline comparison makes this concrete. The two models share only **one** error (the "went 3-11 against top 4 seeds" comment). The LLM got right exactly the comments where the fine-tuned model was fooled by surface style: Yao and Bosh (stat-sounding hot_takes) and Mo Wagner (a one-stat analysis). Judging whether a stat actually proves anything takes basketball knowledge, and a 120B model has it while DistilBERT trained on 105 comments doesn't. The LLM's own errors run the other way: it's *stricter* about evidence, calling three analysis comments hot_takes (e.g., the free-throw-to-3-point-range argument), and it labeled the Grizzlies stats comment analysis where my rule says reaction. That's a fair reading of an edge case I had to write a special rule for.

Against the success criteria in `planning.md` (accuracy ≥ 0.65, macro-F1 ≥ 0.60, every class F1 ≥ 0.50, at least match the baseline), the fine-tuned model **misses** on accuracy (0.565) and macro-F1 (0.571), passes the per-class floor (lowest F1 0.53), and **fails to match the baseline**. It isn't deployable. If I were building a real r/nba tagging tool today I'd use the prompted LLM. Fine-tuning would only be worth it with far more data (likely 1,000+ comments with many stat-flavored hot_takes and calm reactions) or for cost and latency reasons, where a small local model is cheaper to run at scale.

## Spec reflection

- **Where the spec helped:** writing the edge-case decision rules in `planning.md` *before* labeling ("one cherry-picked stat → hot_take", "stat used to vent → reaction") meant the 21 borderline comments were labeled by rule instead of by mood, and gave the error analysis something concrete to test against. Failure 1 is visible as a weakness *in that rule* precisely because the rule was written down.
- **Where implementation diverged:** the spec planned 200+ examples from Reddit directly. Reddit blocked automated access, so data came from a public archive. Collection stopped at 150, which shrank the test set to 23 and made every metric noisy. Training also moved from Colab to a local script, and the notebook's warmup default had to change once the default run turned out to barely train. Finally, the spec's success criterion assumed fine-tuning would at least match the baseline; it didn't, which changed the conclusion from "deploy the fine-tuned model" to "the prompted LLM is the better tool at this data size."

## AI usage


1. **Data collection and pre-labeling.** I directed Claude (Claude Code) to pull r/nba comments and pre-label them with my analysis/hot_take/reaction taxonomy. It fetched comments from the Arctic Shift archive after Reddit refused direct access, filtered out bots and off-topic comments, labeled 150, and flagged 21 borderline cases with notes. Every row is marked `claude_prelabel_needs_review`. 
2. **Pipeline and hyperparameters.** I had Claude run the starter notebook's pipeline locally. It noticed that `warmup_steps=50` exceeds the 21 total training steps, ran the notebook defaults first (loss stayed at chance level), then a corrected configuration selected on validation accuracy. It also caught that one of the baseline prompt's examples was in the test set and replaced all prompt examples with training-set comments. When the baseline run hung, it diagnosed that Groq returns 404 for the Llama 4 Scout model named on the project page and switched to the notebook's `gpt-oss-120b`.
3. **Failure analysis.** Claude grouped the 10 wrong test predictions into patterns (stat vocabulary → analysis, calm reactions → hot_take, short posts failing more often) and compared them with the baseline's 6 errors.

## Running it

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -U transformers datasets accelerate groq python-dotenv scikit-learn matplotlib pandas
.venv/bin/python takemeter.py finetune --run tuned      # ~1 min on Apple Silicon
echo "GROQ_API_KEY=..." > .env                          # your key; .env is git-ignored
.venv/bin/python takemeter.py baseline
.venv/bin/python takemeter.py report --run tuned        # evaluation_results.json + confusion_matrix.png
.venv/bin/python takemeter.py classify "LeBron is washed, he can't guard anyone anymore"
```

The same CSV also works unchanged in the course's starter Colab notebook.
