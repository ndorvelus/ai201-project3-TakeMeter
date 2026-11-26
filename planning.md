# TakeMeter — Planning

## Milestone 1: Community and Labels

### Community

TakeMeter classifies comments from **r/nba**, one of the largest sports communities on Reddit, where every game, trade, and injury produces thousands of comments ranging from stat-backed breakdowns to one-line trash talk. I'm sorting comments into **analysis**, **hot_take**, and **reaction**, because regulars on the sub constantly police exactly this line — "source?", "that's a hot take", and "this is a reaction, not an argument" are everyday replies. Telling a supported argument apart from a confident assertion or an in-the-moment outburst is the distinction that decides whether a comment adds something to the discussion.

### Labels

**analysis** — The comment makes a claim and backs it with specific, checkable evidence (stats, records, history, contracts, or concrete tactical/physical observations) that would still support the point if the opinion framing were removed.

- "Tbf his efficiency is improving in recent games because he shooting more 3s and less midrange jumpers, he's never been an efficient midrange scorer plus the Celtics tend to do better when he's taking at least 10 3s a game historically."
- "This is what small sample sizes look like. Mo Wagner is currently shooting 64% from 3 this year."
- *Unsure:* "Grizz are like 2nd or third in 3pt% and 1st in FT% and I honestly have never seen numbers like that in this franchise's history" — real stats, but the point of the comment is disbelief, not an argument. **Decision: reaction.**

**hot_take** — The comment confidently asserts an opinion, judgment, or prediction without real supporting evidence, or with evidence that is vague or picked for effect rather than reasoned with.

- "Bones isn't a lazy defender, he's just bad. Effort gets you pretty far in the regular season."
- "Danny Green is terrible. Farmar was better, not sure why he only got one shot (so far)"
- *Unsure:* "I'm getting annoyed with these Thunder posts. The team is down to playing legit bums heavy minutes… It's probably a nothingburger." — the tone is a reaction, but the core is a confident prediction. **Decision: hot_take.**

**reaction** — The comment is an immediate emotional response (excitement, anger, amusement, sarcasm) to a specific game, play, clip, or piece of news, with little or no argument.

- "I wish I could upvote this post twice. Shamgod deserves all the respect and flowers in the world"
- "Bout time these 2 teams played comp. Hope they enjoyed their 3 month preseason"
- *Unsure:* sarcastic comments like "Trade your entire roster for Wemby and then trade Wemby for a haul of picks and players? That's honestly brilliant" — sarcasm hides an opinion. **Decision: reaction**, since it responds to a specific idea with mockery rather than a stated claim.

### Hardest edge case

**One stat used for effect (analysis vs. hot_take).** Example: "Number of NBA level players: a lot. Number of people who won the dunk contest 3 times in a row: 1. What McClung did is way more impressive." It cites a fact, so it looks like analysis, but the single stat is chosen to win the argument rather than to reason with.

**Decision rule:** if the evidence would support the claim on its own after removing the opinion framing, label it **analysis**. If the evidence is a single, cherry-picked, or decorative fact sitting under a bold claim, label it **hot_take**. → This example is **hot_take**.

A second rule for the reaction boundary: if a comment includes a stat but its purpose is to express a feeling about something that just happened, label it **reaction**.

### Scope note

Most r/nba comments are not takes about basketball (politics, insults, arguments about the subreddit itself). I only collected comments about basketball, so the "label 90% of posts" requirement applies to basketball comments, not to every comment on the sub.

---

## Milestone 2: Spec

### 1. Community — why r/nba fits a classification task

r/nba is huge and active every night of the season, so there is a steady supply of fresh, public, text-only comments. The discourse is varied in exactly the way this task needs: the same game thread contains box-score breakdowns, confident "X is overrated" claims, and all-caps outbursts about a single play. Those three modes are also how regulars judge each other's comments, so the labels describe a distinction the community already cares about rather than one I invented.

### 2. Labels

Defined above under Milestone 1 (analysis, hot_take, reaction), each with a one-sentence definition, two examples, and one unsure case.

### 3. Hard edge cases and how I handle them

| Boundary | What it looks like | Rule |
|---|---|---|
| analysis ↔ hot_take | A bold claim with one stat underneath it | If the evidence supports the claim on its own with the opinion framing removed → analysis. One cherry-picked or decorative fact → hot_take. |
| analysis ↔ hot_take | Eye-test observations with no numbers ("he drives into a packed paint when he has open 3s") | Specific, observable, checkable-on-film details → analysis. Generic traits ("he's soft", "he's just bad") → hot_take. |
| reaction ↔ analysis | A stat inside an emotional response ("TWENTY offensive boards") | If the purpose is to vent or marvel about something that just happened → reaction. |
| reaction ↔ hot_take | Sarcasm, or an annoyed comment that ends in a prediction | Sarcasm aimed at a specific event or idea → reaction. If the comment's core is a confident claim or prediction → hot_take. |

During annotation I record every case that made me pause in the CSV's `notes` column (prefixed `HARD:`), with the rule I applied, so similar posts get the same label.

### 4. Data collection plan

- **Source:** public r/nba comments. Reddit blocks anonymous automated access, so comments were pulled from the Arctic Shift public Reddit archive. Each row keeps a `source_url` back to the original comment; usernames are not stored.
- **Sampling:** comments from 10 different days across two weeks, so the data isn't dominated by one game or storyline. Filtered out bots, deleted comments, links, very short replies (< 60 characters), and off-topic comments (politics, insults, meta arguments about the subreddit).
- **Target per label:** roughly equal thirds; at minimum 20% per label and no label above 70%.
- **Final dataset:** 150 labeled comments — analysis 51 (34%), hot_take 51 (34%), reaction 48 (32%). This is below the project's 200-example minimum; the trade-off is a smaller test split (23 comments), which makes every metric noisy (one comment ≈ 4.3 accuracy points).
- **If a label were underrepresented:** pull more days from the archive and read specifically for that label (reaction was the hardest to fill, because most emotional comments were off-topic insults rather than responses to basketball events).

### 5. Evaluation metrics

- **Accuracy** for both models, as the headline number — the classes are balanced, so accuracy isn't hiding a majority-class shortcut.
- **Per-class precision, recall, and F1**, because the interesting failure is a specific boundary (I expect analysis ↔ hot_take to be the hardest). A model can reach decent accuracy while never predicting one class correctly; per-class F1 exposes that.
- **Macro-F1** as the single comparison number, because it weights all three labels equally.
- **Confusion matrix**, to see the *direction* of errors (e.g., analysis predicted as hot_take vs. the reverse).
- Because the test set is 23 comments, I'll treat differences of one or two comments between models as noise and say so in the report.

### 6. Definition of success

The fine-tuned model is "good enough" for a real community tool (for example, a bot that tags comments in a discussion thread) if it reaches:

- **Accuracy ≥ 0.65** and **macro-F1 ≥ 0.60** on the test set (random guessing is ~0.33),
- **every class F1 ≥ 0.50** (no label the model effectively can't see), and
- **at least matches the zero-shot Groq baseline** — if a general LLM with no training does as well, fine-tuning hasn't earned its keep.

I'd consider it *useful but not deployable* if it clears the accuracy bar but one class falls below 0.50 F1.

### AI Tool Plan

- **Label stress-testing:** Give an LLM the three definitions and the edge-case rules, and ask for 8 posts that sit on the analysis ↔ hot_take and reaction ↔ hot_take boundaries. Any post I can't label cleanly with my rules means a definition needs tightening before more annotation.
- **Annotation assistance (used):** Claude pre-labeled all 150 comments using the definitions above, and wrote a `HARD:` note on 21 borderline cases. Every row is tracked with `labeled_by = claude_prelabel_needs_review` until I personally re-read and confirm or correct it; I'll change that column once reviewed and disclose the workflow in the README's AI usage section.
- **Failure analysis:** After evaluation, give the list of misclassified test comments (text, true label, predicted label, confidence) to an LLM and ask for recurring patterns — post length, sarcasm, stats inside emotional posts, a single label pair dominating the errors. I'll verify each suggested pattern by re-reading the examples myself and only report patterns that actually hold for most of the cited cases.

---

## Stretch Features Plan (written before starting them)

The main evaluation showed the fine-tuned model at 0.565 test accuracy vs. 0.739 for the zero-shot baseline, with errors that looked style-driven. The 23-comment test set is too small to test patterns or calibration reliably, so the stretch work starts by getting more predictions.

### Shared setup: 5-fold cross-validation
- Re-train the tuned configuration 5 times with stratified 5-fold CV over all 150 comments (inside each fold, 15% of the training part is held out for best-epoch selection). Every comment gets exactly one **out-of-fold** prediction from a model that never saw it.
- These 150 predictions are used **only** for the calibration and error-pattern stretch analyses. The headline test-set numbers in the README stay as they are.

### 1. Confidence calibration
- **Question:** does a higher-confidence prediction actually get it right more often?
- **Method:** bin the 150 out-of-fold predictions by confidence, report accuracy per bin, the expected calibration error (ECE), and a reliability diagram (`calibration.png`).
- **Success:** accuracy rises from the lowest-confidence bin to the highest, and ECE < 0.10. If the model is over- or under-confident, say which.

### 2. Error pattern analysis
- **Hypotheses**, taken from the test-set failures:
  - **H1:** hot_takes that contain numbers or stat vocabulary get misread as analysis.
  - **H2:** reactions without obvious internet-reaction markers (lol, lmao, caps, emoji, !, ?) get missed.
  - **H3:** short comments (< 100 characters) are misclassified more often.
- **Method:** measure each with simple text features over all 150 out-of-fold predictions, comparing error rates with vs. without the feature.
- **Standard for reporting:** a pattern is "systematic" only if the gap holds on the full 150, not just on the 3–4 test examples it came from. Patterns that don't hold get reported as not holding.

### 3. Deployed interface
- **What:** a local Gradio web app (`app.py`): paste a comment, see the predicted label and the confidence for all three labels.
- **Requirements:** it uses the final tuned model and includes example comments for the demo video.
- **Done when:** it runs with one command documented in the README.

### 4. Inter-annotator reliability
- **Who:** a second **person** (not an AI) labels 30 comments independently.
- **The sheet:** stratified (10 per label), shuffled, with my labels hidden. They get only the label definitions and decision rules from this doc.
- **Analysis:** report percentage agreement and Cohen's kappa, plus a table of disagreements and which boundary each falls on.
- **Status:** this can't be completed by AI. The sheet and the scoring script are prepared, and the results wait on a real second annotator.
