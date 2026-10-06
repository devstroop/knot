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

## Frozen holdout

`scripts/gita_retrieval_holdout.jsonl` is a separate 24-case set (12 answerable,
12 unanswerable) with no repeated case IDs or question strings from the
development set above. Its expected citations were checked against the local
Swami Sivananda index. The queries span additional verse topics and new
out-of-domain categories. Keep this file out of threshold/model selection;
the values below were chosen on the development set before evaluating this
holdout. It is still a small, single-author-authored diagnostic, not an
independently adjudicated benchmark.

Run the holdout with the default retriever:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita \
  --author "Swami Sivananda" \
  --eval-set scripts/gita_retrieval_holdout.jsonl \
  --top-k 5
```

The report records the holdout file hash. The current SHA-256 is
`bb6cf56dc9ba1cd5798a0997be0f0beab65c106fd5510bb37496ea47b19a1947`.

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

### Development-set retriever results

On the development set, unthresholded MiniLM at `k=5` achieved answerable-citation
Recall@5 **0.750**, answerable-case Recall@5 **0.917**, MRR@5 **0.757**, and
unanswerable no-evidence rate **0.000**. At cosine cutoff **0.35**, the first
three metrics were unchanged and unanswerable no-evidence rose to **1.000**.
On the same set, default BM25 scored **1.000**, **1.000**, **0.917**, and
**0.417**, respectively; BM25 with at least two distinct query-term overlaps
scored **1.000**, **1.000**, **0.917**, and **0.917**. The semantic model did
not retrieve BG 6.35 for the restless-mind paraphrase, illustrating a remaining
answerable miss. These results are descriptive of this small set only.

### Frozen configuration comparison on holdout

All values below use `k=5` and the same holdout hash above. The BM25
two-term filter, MiniLM `0.35` cosine cutoff, and hybrid weight `0.25`/RRF
constant `60`/`0.015` cutoff were fixed from development-set experiments, not
retuned on this holdout.

| Configuration | Citation Recall@5 | Case Recall@5 | MRR@5 | Unanswerable no-evidence |
|---|---:|---:|---:|---:|
| BM25 default | 0.938 | 1.000 | 0.861 | 0.333 |
| BM25, minimum 2 query-term overlaps | 0.875 | 0.917 | 0.833 | 1.000 |
| MiniLM, no cutoff | 0.750 | 0.917 | 0.778 | 0.000 |
| MiniLM, cosine cutoff 0.35 | 0.750 | 0.917 | 0.778 | 1.000 |
| Hybrid, semantic weight 0.25, RRF k=60, no cutoff | 1.000 | 1.000 | 0.958 | 0.000 |
| Hybrid, same fusion, cutoff 0.015 | 0.938 | 1.000 | 0.958 | 0.667 |

Reproduce the fixed thresholded configurations on the same holdout with:

```bash
python3 scripts/evaluate_gita_retrieval.py --gita-repo ../gita \
  --author "Swami Sivananda" --eval-set scripts/gita_retrieval_holdout.jsonl \
  --top-k 5 --min-query-overlap 2

python3 scripts/evaluate_gita_retrieval.py --gita-repo ../gita \
  --author "Swami Sivananda" --eval-set scripts/gita_retrieval_holdout.jsonl \
  --top-k 5 --retriever semantic --min-score 0.35

python3 scripts/evaluate_gita_retrieval.py --gita-repo ../gita \
  --author "Swami Sivananda" --eval-set scripts/gita_retrieval_holdout.jsonl \
  --top-k 5 --retriever hybrid --semantic-weight 0.25 --rrf-k 60 \
  --min-score 0.015
```

On this holdout, the fixed hybrid cutoff has the same citation recall as
default BM25, a higher MRR, and a higher no-evidence rate, while the uncapped
hybrid retrieves all cited verses but returns passages for every unanswerable
query. MiniLM with its development-set cutoff abstains on all unanswerable
cases, but has lower citation recall and MRR. This is a promising signal for
hybrid ranking, not enough evidence to change defaults: twelve answerable and
twelve unanswerable examples are too few to establish generalization, and the
queries/citations have not been independently adjudicated. Expand and
independently review the holdout before drawing a quality conclusion.

### Development-set threshold ablations

On the development set, an exploratory hybrid sweep with
`--semantic-weight 0.25 --rrf-k 60` and
`--min-score 0.015` returned answerable-citation Recall@5 **1.000**,
answerable-case Recall@5 **1.000**, MRR@5 **0.917**, and unanswerable
no-evidence rate **0.500**. This matches BM25 on the answerable metrics but is
only a small change from its **0.417** unanswerable no-evidence rate; the
development set does not show a meaningful hybrid advantage. Reproduce it
with:

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita --author "Swami Sivananda" \
  --retriever hybrid --semantic-weight 0.25 --rrf-k 60 \
  --top-k 5 --min-score 0.015
```

On the 24-case development set for Swami Sivananda's English translation, BM25
at `k=5` achieved answerable-citation Recall@5 **1.00**, answerable-case
Recall@5 **1.00**, MRR@5 **0.917**, and unanswerable no-evidence rate **0.417**
(five of twelve out-of-domain queries had no positive lexical match). Seven
unanswerable queries still retrieved passages through generic word overlap.
These are descriptive results on a small hand-authored set, not a general
quality estimate; the false positives show that no-evidence abstention alone
does not reliably establish that a query is answerable.

On the development set, an exploratory filter requiring at least two distinct
query terms to overlap each returned passage raised the unanswerable
no-evidence rate to **0.917**
while preserving the other three metrics on this set. Requiring three terms
raised no-evidence to **1.00** but reduced answerable-citation Recall@5 to
**0.450**. Neither threshold is enabled by default: this tiny set is not enough
to tune a production relevance threshold, and the filter is still lexical, not
a semantic retriever. Reproduce the ablations with `--min-query-overlap 2`
and `--min-query-overlap 3`, respectively.

## nqlite hybrid comparison (spike)

As a retrieval-side experiment, the same English eval sets were run through
[nqlite](https://github.com/devstroop/nqlite) hybrid retrieval instead of the
hand-rolled Python indexes. A single long-lived `nql-server --stdio`
subprocess ingested the 701 Sivananda verses (citation, author, text) with
locally computed MiniLM-384 embeddings — the same model as the oio semantic
baseline, since nqlite vectors are bring-your-own by design — then answered a
hybrid query per case (`::bm25` + `vector::similarity`, RRF k=60) at k=5.
Answerable metrics reuse oio's `evaluate_retrieval` through a small index
shim, so the numbers are directly comparable. nqlite ranks every candidate
and always returns its top-k rows (fused RRF scores are never zero), so
unanswerable abstention was measured separately with a BM25-positive
companion query: a case counts as no-evidence only when no returned row has a
positive BM25 score.

```bash
python3 scripts/gita_nqlite_spike.py \
  --gita-repo ../gita \
  --server ../nqlite/target/debug/nql-server
```

The harness is `scripts/gita_nqlite_spike.py`; it needs the FastEmbed
environment from `scripts/requirements-gita-semantic.txt` and a built
`nql-server` binary. Corpus revision `c6fce39`, the same checkout the English
sets were validated against.

| Config | Dev cite / case / MRR / no-ev | Holdout cite / case / MRR / no-ev |
|---|---:|---:|
| oio BM25 default | 1.000 / 1.000 / 0.917 / 0.417 | 0.938 / 1.000 / 0.861 / 0.333 |
| oio hybrid + cutoff | 1.000 / 1.000 / 0.917 / 0.500 | 0.938 / 1.000 / 0.958 / 0.667 |
| oio MiniLM + cutoff | 0.750 / 0.917 / 0.757 / 1.000 | 0.750 / 0.917 / 0.778 / 1.000 |
| nqlite hybrid | 0.900 / 1.000 / 0.875 / 0.000 | 1.000 / 1.000 / 0.917 / 0.000 |

nqlite's answerable ranking is competitive: on the frozen holdout it beats
oio BM25 on citation recall (1.000 vs 0.938) and MRR (0.917 vs 0.861); on the
dev set it trails slightly (0.900 vs 1.000) on two partial citation misses —
BG 6.16 ranked below top-5 for the moderation question, and BG 6.9 for the
friend-and-foe question — while case recall stayed 1.000 in both sets. The gap
traces to tokenization: nqlite BM25 lowercases and splits on
non-alphanumerics with no stopwords and no stemming, so generic words match
and every unanswerable query returns candidates. Abstention is therefore
structurally absent (0.000); any `no_evidence` policy for an nqlite-backed
demo must live client-side, the same way the current demo's zero-score-tie
exclusion does.

This is a comparison, not a migration: the default demo, eval sets, and
thresholds are unchanged, and these small-set numbers do not establish that
either retriever generalizes. The interesting follow-up is a demo that keeps
oio's abstention rule on top of nqlite recall.

## nqlite-backed decision demo

`scripts/gita_nqlite_demo.py` is that follow-up: the same typed-decision
contract as `gita_decision_demo.py` (`choice`/`score`/`noul` over retrieved
excerpts with citations, same OIO request shapes via the shared builders),
but retrieval comes from nqlite hybrid queries instead of the hand-rolled
indexes. Ranking uses the full query text, exactly as the spike measured;
abstention uses a companion BM25 query over the query's content terms only
(English stopwords stripped, unstemmed), which reproduces oio's own
stopword-aware no-evidence semantics — measured at 0.417 dev / 0.333
holdout abstention, identical to oio BM25's 0.417 / 0.333. A query with no
content terms abstains without starting the server. Retrieval plumbing
(server ownership, ingest, row parsing) is shared with the spike script.

```bash
OIO_MODEL_DIR=/path/to/laya-english cargo run -p oio-serve  # port 8000
python3 scripts/gita_nqlite_demo.py \
  "How should someone act without becoming attached to the result?" \
  --gita-repo ../gita \
  --server ../nqlite/target/debug/nql-server \
  --mode choice --top-k 5
```

The response carries the same `decision` / `retrieved_passages` /
`no_evidence` / `corpus` shape plus a `retriever` block (engine, hybrid mode,
evidence rule, BM25 max score, embedding model). Unit tests use a fake server
and never touch a real binary: `python3 -m unittest discover -s scripts -p
'test_gita_nqlite_demo.py' -v` (10 tests: evidence rule, abstention and
insufficient-candidate skips, payload shapes, citation mapping, CLI
validation).

Smoke observation with the existing English checkpoint (`55cf4c4...`):
nqlite retrieved BG 3.5/3.19/3.25/3.26/18.23 for the attachment question and
OIO chose BG 3.5 at confidence ~0.25; score mode returned 2.63 ("relevant
support but gaps"); noul returned true at 0.749; and a Bitcoin-price query
abstained with BM25 max score 0.0 and no OIO call. As with the BM25 demo, the
checkpoint is not a qualified Gita reranker — confidences are low and the
ranking comes from the retriever. The live demo path needs the FastEmbed
environment (`scripts/requirements-gita-semantic.txt`); the unit tests do not.

## Hindi holdout

The authorized clone also carries Hindi translations, so the retrieval
diagnostic was extended to a second language with
`scripts/gita_retrieval_holdout_hi.jsonl`: 24 balanced cases (12 answerable,
12 unanswerable) with original Hindi questions, disjoint case IDs and query
strings from both English sets. Expected citations were validated against the
local Swami Tejomayananda index (701 verses); no question text or verse text
was copied from another dataset.

```bash
python3 scripts/evaluate_gita_retrieval.py \
  --gita-repo ../gita \
  --language hindi \
  --eval-set scripts/gita_retrieval_holdout_hi.jsonl \
  --top-k 5
```

The report records the language, author, and corpus hashes. The current
holdout SHA-256 is
`8771c21ae19fc7ba423090fd6f4ee8c622ece729ed91ead8d2ff04f74c6b9ac9`.

Two implementation notes are load-bearing for these numbers:

- Python's `\w` excludes combining marks, so the previous token pattern split
  Devanagari words at every vowel sign and virama. Tokenization now keeps
  word characters plus Devanagari/transliteration combining marks, and the
  Hindi function-word list (postpositions, pronouns, auxiliaries, question
  words) is treated as stop words. English tokenization is unchanged: all 701
  English passages tokenize identically before and after this change, and the
  English development and holdout metrics above are unaffected.
- `sentence-transformers/all-MiniLM-L6-v2` is English-only, so
  `--language hindi` accepts BM25 only; semantic and hybrid retrieval are
  rejected with an error rather than silently producing meaningless
  embeddings.

Results at `k=5` with default BM25:

| Configuration | Citation Recall@5 | Case Recall@5 | MRR@5 | Unanswerable no-evidence |
|---|---:|---:|---:|---:|
| BM25 default | 0.929 | 1.000 | 1.000 | 0.167 |
| BM25, minimum 2 query-term overlaps | 0.929 | 1.000 | 1.000 | 0.667 |
| BM25, minimum 3 query-term overlaps | 0.929 | 1.000 | 1.000 | 0.917 |

Every answerable case had its first cited verse at rank 1; the single
citation-level miss is BG 3.8, which ranks below `k=5` for the two-citation
`hi-attachmentless-duty` case. Abstention is much weaker than on the English
holdout (0.167 versus 0.333): ten of twelve unanswerable queries still
retrieve passages from generic-word overlap, including the deliberate traps.
This mirrors the English finding that a lexical match is not evidence of
answerability, and it is a reason to keep Hindi abstention untrusted.

These numbers are descriptive of a small, single-author, same-process
diagnostic set. The Hindi stop-word list was finalized while inspecting
false-positive matches on this set, so this holdout is not fully untouched for
tokenization choices: treat it as a smoke diagnostic and author an independent,
reviewed Hindi test set before making any Hindi quality claim. The live demo
remains English-only.

## Scope and limitations

This is a prototype for measuring the value of a domain corpus with OIO, not a
claim that the model has learned the Gita. BM25 is lexical, not semantic, and
the existing decision checkpoint may choose or score passages poorly. Inspect
retrieved passages and citations alongside each answer. For a quality
evaluation, create a separately reviewed question set, include unanswerable
questions, and report retrieval recall separately from OIO decision quality.

The live demo uses English translation text only; Hindi is supported by the
retrieval evaluation (`--language hindi`) described above. The source clone is
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
