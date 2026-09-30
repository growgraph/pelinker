# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Class views** (`pelinker/kb/classes.py`) decide what counts as one class of the KB,
  without editing the KB. Each is a projection of the pairs KB's own declarations:
  - `raw`: the matched label;
  - `rel`: the canonical relation;
  - `reldir`: the canonical relation plus the mention's direction relative to it.

  In `reldir`, a passive mention is `(R, inverse)` whether or not the KB holds a converse
  entry for `R`. Under raw labels it was its own class in one case and shared the active
  class in the other.

  `add_view_columns` adds `relation`, `relation_direction` and `entity_class` to a mention
  frame. It refuses labels the KB does not hold. Judgements about the KB (which entries
  mean the same thing) are deliberately not accepted, so they cannot reach training
  labels. `canonical_id_map`, `symmetric_ids` and `require_canonical_kb` moved here;
  `pelinker.eval.kb` re-exports them.
- `class_view` / `class_kb_path` on `ClusteringOptimizationConfig` and `LinkerFitConfig`;
  `--class-view` / `--class-kb-path` on `pelinker-model-selection`,
  `pelinker-dim-selection` and `pelinker-scale-curve`; and `class_view` / `class_kb_path`
  on `pelinker-fit`, where the KB path defaults to `kb_path`.
  - `load_selection_frame` applies the view.
  - The grid ARI (`evaluate_cluster_size_grid`) and the fit ARI
    (`compute_clustering_fit_metrics`) score against `entity_class` when it is present.
  - A view enters the search checkpoint fingerprint, with the KB by content hash. A `raw`
    run keeps the fingerprint it had before.
- A direction per cluster.
  - Under a class view, the fit's catalog composition is read between canonical
    relations: the two voices of one relation are one entity whose clusters differ in
    direction, not two entities a cluster merged.
  - Each catalog cluster carries `direction_mix` and `dominant_direction`
    (`kb_out.annotate_cluster_directions`, `cluster_direction_summary`).
  - `kb_out_to_reldir_map` returns (input-KB id, direction) per KB-out id.
  - `Linker.cluster_direction` holds the map, and `Linker.predict` and `/link` emit
    `direction_predicted` for a linker fitted with a view.
  - `/info` reports `emits_direction`.
  - The catalog's fit provenance records `class_view` and `class_kb_sha256`.
  - Mention `direction` and the view columns now travel into
    `Linker.training_cluster_frame`.
- `docs/user_guide/api.md`: the HTTP routes, request fields and entity-row fields,
  including how a KB-out id reaches the input KB.
- `run/analysis/weak_label_check.py` reports classes per view and the mention mass by
  direction relative to the canonical relation. A label the KB does not hold is now a
  violation.
- Gold evaluation pipeline (`run/eval/`, `pelinker/eval/`): document sampling with
  `primary` / `double` / `reserve` roles, LLM pre-annotation, a human review round trip
  with Cohen's κ, and reference baselines (lexical lemma, sentence-encoder k-NN, an LLM
  linker, and a fitted artifact) scored through one harness in two regimes — end-to-end
  span+link, and linking-only over gold spans. Documented in
  `docs/user_guide/evaluation.md`.
- `eval` extra carrying the LLM provider SDKs. The linker itself never calls an LLM; only
  the evaluation path does.
- `openai` joins `gemini` and `anthropic` in the `pelinker.eval.llm` provider seam. The
  agreement slice wants a second model *family*, and the LLM baseline a third model that
  did not annotate the gold, so one SDK was never enough. Credentials stay per provider
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`) and the
  response cache is keyed on the provider, so one provider's answer is never replayed for
  another.
- `run/preprocessing/derive_inverse_pairs.py` links converse pairs across the whole KB —
  declared `owl:inverseOf` plus three surface-form rules, each recorded in
  `inverse_source` — and elects a canonical member per pair (`is_canonical`,
  `canonical_entity_id`), so a passive mention has exactly one valid encoding and
  orientation lives in `direction`.
- A curator review loop for those derived pairs: unreviewed proposals are exported to
  `<output>.pending_review.csv`, and `--review-csv` applies the verdicts — `accept` clears
  the flag, `reject` unlinks the pair so each member is its own canonical form again. An
  unreviewed wrong pair merges two distinct relations onto one id, and prompts, gold and
  scores would then all agree with the mistake.
- Symmetric relations: `extract_properties_ro.py` emits `is_symmetric` (declared
  `owl:SymmetricProperty`, or defined by the chain `inverse(P) ∘ P`), and the pairs KB
  carries it. The surface-form rules never pair a symmetric entry — "connected to" looks
  passive but has no converse. The annotation prompt marks symmetric labels, and
  `annotate_llm.py` sets an oriented answer on one to `symmetric`, flagged
  `direction_coerced`. Entries RO does not cover are marked symmetric through a curated
  list (`derive_inverse_pairs.py --symmetric-csv`, default `data/curated/symmetric.csv`).
- The passive surface rule recognises irregular participles ("bound by" ↔ "binds to"),
  not only `-ed` / `-en` forms.
- The curated KB carries converse entries (`X-ed by`) for the GeneWays verbs, so a
  passive mention matches its own entry and folds onto the active relation with
  `direction=inverse`; `stimulates` and `interrupts` are restored. The converse pairs go
  through the curator review loop like every other derived pair.
- `score_predictions_against_ground_truth` and both harness regimes take `canonicalize`,
  applied to predicted **and** gold ids after the KB-out → KB-in bridge.
- `reasoning_effort` in `pelinker.eval.llm.complete` (openai only) and
  `annotate_llm.py --reasoning-effort`. It is part of the cache key only when set, so
  existing caches replay unchanged, and it is added to the default annotator tag
  (`<model>@<effort>`).
- `annotate_llm.py` accepts an object wrapping the requested array
  (`{"annotations": [...]}`), since object-only JSON modes produce that shape. It also
  reports `duplicate_occurrence`, a mention re-emitted after every occurrence is taken,
  separately from `surface_not_found`, quoted text absent from the abstract.

### Fixed

- Truncated LLM answers were cached as final. A response cut off at the output limit
  would replay on every later run as an unparsable document with no candidates, and
  raising the limit would not retry it. Reasoning models hit this because their hidden
  reasoning counts against the output limit. `pelinker.eval.llm` now raises
  `LLMIncompleteError` and caches nothing when the provider reports truncation (openai
  `status`, gemini `MAX_TOKENS`, anthropic `max_tokens`) or returns empty text.
  `annotate_llm.py` reports those documents as `truncated` and leaves them out of the
  output, so they are retried on the next run.

- Corpus embedding (`embed_kb_corpus`, `pelinker-fit` stage A) crashed at the second read
  buffer with "generator already executing": the stream that re-attaches the peeked
  first chunk closed over a variable later rebound to that same stream. Any input longer
  than one `input_buffer_rows` pass was affected. The stream is now built by a
  module-level `prepend_first`.

- Agreement (κ) is computed over the documents both annotators covered. It previously
  walked every document of the first annotator, so with a second annotator on the `double`
  slice alone — the normal case — each hit on an unannotated document counted as a
  disagreement and κ collapsed for a reason unrelated to agreement. The report now carries
  `n_docs_shared`, `n_only_a` and `n_only_b`.
- κ is symmetric: a span only the second annotator marked now enters as an explicit `∅` on
  the first annotator's side, so missing a mention and inventing one are penalized alike.
  The docstring had claimed this; only one direction was implemented.
- The review sheet surfaces spans only the second annotator proposed (`origin=other`), and
  `import` adopts the accepted ones from `--other`. Verified gold could previously never
  exceed the first annotator's recall.
- Review rows for documents the second annotator never saw read `not-covered` instead of
  an empty cell that looks like a disagreement.
- Baseline ids are scored in canonical space. Gold carries the canonical member of each
  converse pair while the baselines choose from the whole KB, so a correct "regulated by"
  link was scored as an error — measuring the vocabulary's redundancy rather than the
  system.
- The linking-only regime applies the KB-out → KB-in bridge, so a fitted artifact's minted
  ids are comparable there as they already were end-to-end.
- `.gitignore` no longer excludes the package source directory `pelinker/reports/`. The
  entry was the unanchored `reports/`, which matches a directory of that name at **any**
  depth, so four modules that ten others import were never in the repository and a fresh
  clone could not import `pelinker` at all. The pattern is now anchored to the repo root,
  where measurement output lives.
- `pelinker/reports/` and `pelinker/text/` gained `__init__.py`. As implicit namespace
  packages they imported fine but could not be collected by mkdocstrings, so
  `mkdocs build --strict` failed and their modules had no API reference pages.
- `run/loop.embed.kb.corpus.sh` invokes `uv run python` rather than bare `python`, so the
  batch grid uses the locked environment like every other entry point.
- Refusing a KB that predates the pair derivation now names the file that was passed, says
  that a `*.inverse.csv` is the derivation's *input*, and points at the sibling
  `*.pairs.csv` when one exists. Several usage blocks had shown exactly the KB the check
  rejects.

### Security

- Raised `transformers` to `>=5.5.0,<6`, closing GHSA-29pf-2h5f-8g72 (RCE, patched in
  5.3.0), GHSA-fgcw-684q-jj6r (arbitrary code execution in the LightGlue model loading
  path, patched in 5.5.0) and GHSA-69w3-r845-3855 (arbitrary code execution in `Trainer`).
- Raised the `torch` floor to `>=2.13.0` (GHSA-rrmf-rvhw-rf47, memory corruption in
  `torch.jit.script`), and relocked `pillow` 12.3.0 (13 advisories) and `setuptools`
  83.0.0 (GHSA-h35f-9h28-mq5c).
- Added `.github/dependabot.yml` (weekly `uv` and `github-actions` updates); there was no
  Dependabot configuration before.

### Changed

- **`pelinker-fit` and the selection CLIs default to `class_view=reldir`** and therefore
  need the pairs KB (`properties.synthesis.2.pairs.csv`): as `kb_path` for the fit, or as
  `--class-kb-path` for the searches. Pass `class_view=raw` / `--class-view raw` to
  reproduce earlier fits and searches exactly. Hyperparameters selected against raw labels
  do not carry over to a `reldir` fit, because the objective's reference partition
  differs.

- The gold sampler (`run/eval/sample_gold.py`) draws only eligible abstracts:
  publication year, length and sentence bounds, English, not a duplicate, and not in the
  fit corpus (`--exclude-corpus`). Each dropped row is counted by reason. It stratifies on
  mentions detected with the fit's own definition (`pelinker.text.mentions`) rather than
  raw lemma-window counts, which mostly tracked abstract length. Tail strata (rare labels)
  are over-allocated. Each document records its inclusion probability `p_incl`, and the
  sample grows in batches with `--extend` (stable ids, roles per batch).
  `run/eval/audit_sample.py` re-checks a sample and reports KB coverage per batch.
- Gold hits carry `surface_anchored`: `false` marks a paraphrase ("leads to" for
  `causes`), which is kept and scored as its own slice. `gold_review_sheet.py import`
  records each hit's `verdict` and writes `<output>.verification.json`: accept, fix and
  reject counts and the correction rate, split by anchored and paraphrase.
  `audit_sample.py` reports the paraphrase share and the verification summary.
- `run/analysis/weak_label_check.py`: acceptance check for a stage-(A) parquet. It fails
  on multi-labelled or nested spans, on relation/converse collisions, and on a symmetric
  relation with an inverse direction.
- `run/loop.embed.kb.corpus.sh` refuses to overwrite an existing parquet. It adds
  `--models`, `--layers`, `--max-input-buffers` and `--no-gpu`, fails fast, and warns when
  the KB has no `is_symmetric` column.
- `annotate_llm.py` rejects hits on verb-predicate labels whose span has no verb
  (`non_verbal`), after the response cache (`--verbal-only`, on by default).
- Weak supervision labels verb predicates from the dependency parse instead of lemma
  windows (`pelinker/text/predicates.py`). Each verb mention gets at most one KB label,
  chosen by voice: a passive mention takes the KB's converse entry ("activated by"), even
  with words between the verb and its agent or no agent at all, and falls back to the
  active entry with `direction=inverse` when the KB has no converse. Symmetric relations
  are never flipped; nouns ("controls") and prenominal modifiers without an object
  ("activated T cells", "binding sites") are not labelled. A label's preposition must
  attach to the verb — "developed complications" is not `develops from` — except the
  agent of a passive `by` label. The embedded span is the verb, extended to an adjacent label
  adverb ("negatively regulated") — the same kind of contiguous window prediction
  proposes, so prediction stays embedding-only. Non-verb labels ("part of") keep lexical
  matching, and every mention site keeps one label. Mention parquets gain `direction` and
  `surface_rule`. Hyperparameters selected on mentions extracted before this change were
  fitted to contradictory labels and should be re-selected.

- **BREAKING:** `min_cluster_size` selection is rebuilt. The DBCV+ARI objective was an
  arithmetic mean of two per-curve **min–max normalized** metrics, and min–max normalizes
  *range*, not noise: a metric that barely moved got its noise rescaled to match the other
  and then handed 50% of the objective. The combined std was also computed from the *raw*
  metrics while the mean used the normalized ones, so every consumer of that std — the
  `lower_bound` discount, the smoothing weights, the leaderboard tie-break, the persisted
  `score_std_at_chosen` — was mixing units.
  - `grid_objective` is now `dbcv_ari_geomean`: `sqrt(max(dbcv, 0) * max(ari, 0))`. Its
    ranking is invariant to rescaling either metric, so no normalization step is needed at
    all, and it is conjunctive — a high DBCV can no longer buy off a near-zero ARI.
    `dbcv_ari_mean_minmax` and `dbcv_ari_mean_raw` are removed.
  - Metrics are combined **per bootstrap sample and then pooled**, not pooled and then
    combined, so DBCV/ARI correlation is handled correctly and the samples stay paired.
  - The plateau/derivative selector is replaced by a one-standard-error rule using *paired*
    standard errors: take the argmax, then move to the largest `min_cluster_size` reachable
    through consecutive grid points within `grid_one_se_k` (default 1.0) standard errors of
    it. Uncertainty can now only break a statistical tie; it can never promote a point with
    a worse mean, which `mean - 1*std` did. `optimization_method`, `grid_plateau_fraction`
    and `grid_derivative_rel_tol` are removed; `grid_one_se_k`,
    `grid_one_se_contiguous` and `grid_one_se_min_samples` replace them.
  - The rule is skipped below `grid_one_se_min_samples` (default 6) bootstrap samples. A
    standard error estimated from three numbers is mostly noise; measured by
    leave-one-sample-out on this repo's grid exports, acting on it made the choice *less*
    reproducible, and the slide only starts paying for itself around 6–8 samples.
  - The rule narrows the leave-one-sample-out spread of the chosen `min_cluster_size`;
    measure it on your own grid exports with `pelinker-replot --grid-one-se-k`.
- **BREAKING:** the outer candidate ranking uses the same clipped geometric mean as the
  grid. It previously min–max normalized **across the leaderboard**, so adding an
  irrelevant candidate could reorder the good ones; scores are now computed per row and are
  comparable across runs. `attach_outer_scores` / `pick_best_row` lose `use_minmax`, and
  `CHECKPOINT_VERSION` is bumped to 2 because `singleton_scores_by_key` persists these
  values and a resumed v1 checkpoint would mix the two scales.
- Search summaries read **ARI at the pooled `min_cluster_size`**. DBCV was already read
  there, but ARI came from each sample's own best size, so `outer_score` combined two
  metrics measured at different hyperparameters.
- The combined-objective panel now renders in real runs. Runners passed
  `chosen_min_cluster_size` and no `grid_solve`, which suppressed the solve entirely, so
  the fourth axis shipped empty and titled "Grid objective (unavailable)" in every run
  except `pelinker-replot`. The panel also gained the ±1 SE band, the decision boundary,
  the tied grid points and the argmax marker; single-sample `plot_metrics` gained the panel
  and the chosen-value line; and the DBCV-vs-ARI scatter now draws the Pareto front.
- `pelinker-replot` gained `--grid-one-se-k` and no longer resets unrelated solver knobs to
  their defaults when only `--grid-cluster-count-reward` / `--grid-n-entities` are given.
- **BREAKING:** `pelinker`'s flat module layout is reorganised into domain subpackages —
  `core/`, `text/`, `data/`, `embed/`, `clustering/`, `screener/`, `linker/`, `kb/`,
  `search/`, `reports/`. Every import path under `pelinker.*` changed; the console-script
  entry points did not.
- `pelinker.analysis` is gone. It fused three unrelated concerns, now split into
  `text/lexical.py`, `screener/evaluation.py`, `clustering/metrics.py`,
  `data/frames.py` and `search/grid_solver.py`.
- `pelinker.io` is now `pelinker.data` — the old name shadowed the stdlib `io` module
  inside the package namespace.
- `pelinker.reporting` is split into `reports/schema.py` (the dataclasses),
  `reports/paths.py` (basenames and path builders), `reports/io.py` (JSON read/write)
  and `reports/summary.py` (flat-row aggregation).
- `pelinker.util` is split into `text/models.py`, `text/tokenize.py`,
  `text/chunking.py` and `text/embed.py`, plus `core/paths.py` and `kb/registry.py`.
- `ChunkMapper` and `ReportBatch` move from `core/onto.py` into `text/`. They called into
  the text pipeline from inside their methods purely to dodge the `util` ⇄ `onto` import
  cycle; that cycle is now gone and `core/onto.py` imports nothing from `pelinker`.
- **BREAKING:** the default spaCy pipeline is now `en_core_web_lg` instead of
  `en_core_web_trf`. `en_core_web_trf` requires `spacy-transformers`, which pins
  `transformers<4.53.3` and therefore cannot coexist with the patched `transformers`
  releases above. pelinker uses spaCy only for tokenization, `lemma_`, `tag_` and `pos_`,
  so a non-transformer pipeline covers the required surface — but POS/lemma accuracy
  differs, so re-score against `data/ground_truth` before relying on existing thresholds.
- Raised `sentence-transformers` to `>=5.6,<6`, the first release permitting
  `transformers` 5.x.

### Fixed

- `matplotlib`, `seaborn`, `plotly` and `pySankey` are imported at module level by
  library code but were declared only in the `dev` extra, so four of the seven console
  scripts (`pelinker-model-selection`, `-dim-selection`, `-replot`, `-scale-curve`)
  raised `ModuleNotFoundError` on a plain install. They are runtime dependencies now.
- The `dev` extra pins the `en_core_web_lg` wheel instead of relying on
  `spacy download`, which any subsequent `uv sync` prunes — silently re-skipping every
  test that needs the model.
- `ruff` was not reporting unused imports (no `[tool.ruff.lint]` section, so the
  pre-commit defaults applied); ~100 had accumulated. Now configured and cleaned.
- `pelinker-serves` segfaulted on import. TensorFlow (pulled in via `tf-keras` for
  ParametricUMAP) and torch's bundled `triton` conflict at the native level: whichever
  order left `triton.runtime` importing after TensorFlow crashed the interpreter.
  `pelinker/__init__.py` now imports torch and `triton.runtime` first.
- `text_to_tokens_embeddings` used `tokenizer.batch_encode_plus`, which was removed in
  transformers 5.x; it now uses the standard `tokenizer(...)` call. This path was not
  covered by CI because the tests exercising it skip when the spaCy model is absent.

### Removed

- `mypy` from the `dev` extra and the unused `.pylintrc` — neither was wired into
  pre-commit or CI.
- `pip` and a duplicate `cupy-cuda12x` from the runtime dependencies. `cupy` is not
  imported anywhere in the package; it remains available through the `gpu` extra.
