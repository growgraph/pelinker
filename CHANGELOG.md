# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - Measured leave-one-sample-out spread of the chosen `min_cluster_size`, old rule → new:
    2.60 → 1.59 on `dim_selection_2` (35 combos), 0.79 → 0.00 on `clustering-d-sample-3`,
    5.17 → 4.98 on `clustering-pca-dbcv-c-sample-test`.
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
  `search/`, `viz/`. Every import path under `pelinker.*` changed; the console-script
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
