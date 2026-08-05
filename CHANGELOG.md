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

- **BREAKING:** the default spaCy pipeline is now `en_core_web_lg` instead of
  `en_core_web_trf`. `en_core_web_trf` requires `spacy-transformers`, which pins
  `transformers<4.53.3` and therefore cannot coexist with the patched `transformers`
  releases above. pelinker uses spaCy only for tokenization, `lemma_`, `tag_` and `pos_`,
  so a non-transformer pipeline covers the required surface — but POS/lemma accuracy
  differs, so re-score against `data/ground_truth` before relying on existing thresholds.
- Raised `sentence-transformers` to `>=5.6,<6`, the first release permitting
  `transformers` 5.x.

### Fixed

- `text_to_tokens_embeddings` used `tokenizer.batch_encode_plus`, which was removed in
  transformers 5.x; it now uses the standard `tokenizer(...)` call. This path was not
  covered by CI because the tests exercising it skip when the spaCy model is absent.

### Removed

- `mypy` from the `dev` extra and the unused `.pylintrc` — neither was wired into
  pre-commit or CI.
- `pip` and a duplicate `cupy-cuda12x` from the runtime dependencies. `cupy` is not
  imported anywhere in the package; it remains available through the `gpu` extra.
