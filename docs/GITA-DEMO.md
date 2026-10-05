# Gita retrieval + OIO decision demo

This optional local demo uses the authorized clone at
[itsalfredashu/gita](https://github.com/itsalfredashu/gita) as a small,
attributed knowledge source. It demonstrates retrieval followed by an OIO
typed decision; it does **not** train a model or make the full corpus part of
OIO's runtime artifacts.

The first version indexes the English translation by Swami Sivananda and uses
simple lexical BM25 retrieval. OIO then makes one of three decisions over the
retrieved evidence:

- `choice`: select the most relevant cited verse;
- `score`: rate how directly the passages support an answer;
- `noul`: decide whether the passages provide enough evidence without
  unsupported additions.

The demo prints the answer with chapter/verse citation, translation author,
retrieval score, source repository revision, and hashes for the source data.
It sends only the query and retrieved excerpts to OIO. It does not retrieve or
send the full commentary archive.

When BM25 finds no positive-scoring passage, the demo returns
`"no_evidence": true` with `"decision": null` and does not call OIO. This
prevents a zero-score tie from being presented as a meaningful candidate
choice. In `choice` mode, fewer than two positive candidates also produces no
decision and does not call OIO (`"insufficient_candidates": true`); `score` and
`noul` can evaluate a single retrieved passage.

## Run

1. Start OIO with a locally available checkpoint:

   ```bash
   OIO_MODEL_DIR=/path/to/laya-english cargo run -p oio-serve
   ```

2. From the OIO repository root, ask a question:

   ```bash
   python3 scripts/gita_decision_demo.py \
     "How should someone act without becoming attached to the result?" \
     --gita-repo ../gita
   ```

   Select another primitive or number of retrieved candidates with
   `--mode score|noul` and `--top-k 3`. Change the retrieval author with
   `--author "Swami Adidevananda"`; available English translation authors are
   validated against the local corpus. The default server URL is
   `http://127.0.0.1:8000`; override it with `--oio-url`.

The retriever and request-shape tests use Python's standard library:

```bash
cd scripts
python3 -m unittest -v test_gita_decision_demo.py
```

## Retrieval evaluation

Run the small, hand-authored retrieval set without calling OIO:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita \
  --author "Swami Sivananda" \
  --top-k 5
```

Use `--min-query-overlap 2` to reproduce the simple overlap-filter ablation
below. This option affects retrieval evaluation only; the live demo continues
to use the default BM25 behavior.

The set in `scripts/gita_retrieval_eval.jsonl` has twelve answerable questions
with reviewed relevant citations, including paraphrases and several topics,
plus twelve intentionally out-of-domain questions. The latter include
generic-word-overlap challenges (for example, finance, weather, and software
queries) as well as queries with little expected lexical overlap. Its JSON
report includes answerable citation Recall@k, answerable case Recall@k, MRR@k,
and the rate at which unanswerable queries produce no positive lexical match.
It reports retrieval only; these numbers do not measure OIO's decision
quality. This remains a small hand-authored diagnostic set, not a representative
or independently validated benchmark. A nonzero lexical hit for an
unanswerable question is a retrieval false positive, not evidence that the
corpus answers it.

BM25 remains the default baseline. An optional local semantic baseline uses
FastEmbed's ONNX Runtime implementation of
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The model card declares Apache-2.0; FastEmbed lists the model as 384-dimensional
with a 256-token truncation length. Install the pinned optional dependency:

```bash
python3 -m venv .venv-gita-semantic
. .venv-gita-semantic/bin/activate
python3 -m pip install -r scripts/requirements-gita-semantic.txt
```

Run semantic retrieval locally against the same cases:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita \
  --author "Swami Sivananda" \
  --retriever semantic \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5
```

The evaluator also supports reciprocal-rank fusion of BM25 and MiniLM:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita \
  --author "Swami Sivananda" \
  --retriever hybrid \
  --semantic-weight 0.5 \
  --rrf-k 60 \
  --top-k 5
```

`--semantic-weight` controls the semantic share from 0 (BM25 ranks only) to 1
(semantic ranks only). Reciprocal-rank fusion avoids mixing BM25 and cosine
score scales. The fused score is rank-derived and still needs its own cutoff
evaluation; BM25 and cosine cutoffs are not interchangeable.

FastEmbed downloads the public ONNX model on first use; embedding and retrieval
then run locally. No query, corpus passage, or API credential is sent to a
hosted inference service. The optional `--min-score` is a cosine-similarity
cutoff for semantic retrieval (and a raw BM25-score cutoff for BM25); scores
are retriever-specific and must not be compared across retrievers. For example:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita --author "Swami Sivananda" \
  --retriever semantic --top-k 5 --min-score 0.35
```

With no cutoff, semantic search always returns its nearest passages, so
unanswerable no-evidence rate is zero. Sweep cutoffs on this diagnostic set to
inspect the recall/abstention tradeoff, but do not treat a threshold tuned on
these cases as production-calibrated.

Retrieval results should be compared before evaluating OIO decisions. The
hand-authored set is a small diagnostic, not a representative or independently
validated benchmark.

On this set, unthresholded MiniLM at `k=5` achieved answerable-citation
Recall@5 **0.750**, answerable-case Recall@5 **0.917**, MRR@5 **0.757**, and
unanswerable no-evidence rate **0.000**. At cosine cutoff **0.35**, the first
three metrics were unchanged and unanswerable no-evidence rose to **1.000**.
On the same set, default BM25 scored **1.000**, **1.000**, **0.917**, and
**0.417**, respectively; BM25 with at least two distinct query-term overlaps
scored **1.000**, **1.000**, **0.917**, and **0.917**. The semantic model did
not retrieve BG 6.35 for the restless-mind paraphrase, illustrating a remaining
answerable miss. These results are descriptive of this small set only.

An exploratory hybrid sweep with `--semantic-weight 0.25 --rrf-k 60` and
`--min-score 0.015` returned answerable-citation Recall@5 **1.000**,
answerable-case Recall@5 **1.000**, MRR@5 **0.917**, and unanswerable
no-evidence rate **0.500**. This matches BM25 on the answerable metrics but is
only a small change from its **0.417** unanswerable no-evidence rate; this set
does not show a meaningful hybrid advantage. Reproduce it with:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita --author "Swami Sivananda" \
  --retriever hybrid --semantic-weight 0.25 --rrf-k 60 \
  --top-k 5 --min-score 0.015
```

On the expanded 24-case set for Swami Sivananda's English translation, BM25
at `k=5` achieved answerable-citation Recall@5 **1.00**, answerable-case
Recall@5 **1.00**, MRR@5 **0.917**, and unanswerable no-evidence rate **0.417**
(five of twelve out-of-domain queries had no positive lexical match). Seven
unanswerable queries still retrieved passages through generic word overlap.
These are descriptive results on a small hand-authored set, not a general
quality estimate; the false positives show that no-evidence abstention alone
does not reliably establish that a query is answerable.

An exploratory filter requiring at least two distinct query terms to overlap
each returned passage raised the unanswerable no-evidence rate to **0.917**
while preserving the other three metrics on this set. Requiring three terms
raised no-evidence to **1.00** but reduced answerable-citation Recall@5 to
**0.450**. Neither threshold is enabled by default: this tiny set is not enough
to tune a production relevance threshold, and the filter is still lexical, not
a semantic retriever. Reproduce the ablations with `--min-query-overlap 2`
and `--min-query-overlap 3`, respectively.

## Scope and limitations

This is a prototype for measuring the value of a domain corpus with OIO, not a
claim that the model has learned the Gita. BM25 is lexical, not semantic, and
the existing decision checkpoint may choose or score passages poorly. Inspect
retrieved passages and citations alongside each answer. For a quality
evaluation, create a separately reviewed question set, include unanswerable
questions, and report retrieval recall separately from OIO decision quality.

The current demo uses English translation text only. The source clone is
treated as authorized for this prototype as requested; author attribution and
source hashes are retained so provenance remains visible. Verify any
per-translation or commentary rights before redistributing corpus text or
using it in a released training artifact. The demo output is not religious
guidance and does not replace interpreting the source in context.

## Initial local smoke observation

With the existing English checkpoint (`55cf4c4...`), the query
"How should someone act without attachment to results?" retrieved BG 3.25 and
BG 2.47 as its two highest BM25 passages. OIO selected BG 18.23 from the
five-candidate list, with answer confidence about 0.22. The confidence is low
and its choice did not match the lexical rank-1 passage. This is useful evidence
that the integration works, but also that the current general checkpoint is
not a qualified Gita reranker. Do not interpret the smoke output as accuracy
or tune thresholds from this single example.
