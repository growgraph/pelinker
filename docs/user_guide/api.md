# HTTP API

`pelinker-serves` (`pelinker.cli.server`) loads one fitted linker and serves it over
FastAPI. The default port is **8599**. Configuration uses Hydra overrides (`host`, `port`,
`model_file_spec`, `thr_score`, `use_gpu`, `cors_allow_origins`); defaults are in
`pelinker/conf/server.yaml`. Interactive docs are at `/docs` while the server runs.

```bash
uv run pelinker-serves model_file_spec=<models>/pelinker.pubmedbert.run1 port=8599
```

## Routes

| Route | Method | Returns |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` |
| `/info` | GET | Loaded artifact: path, embedding metadata, spaCy pipeline, KB config, vocabulary size, cluster count, transform config, `emits_direction` |
| `/model` | GET | Embedding metadata and spaCy pipeline name |
| `/link` | POST | Linked predicate mentions |
| `/link/debug` | POST | The same, plus per-mention diagnostics |

## `POST /link`

Request body — `text` or `texts` is required:

| Field | Type | Meaning |
|---|---|---|
| `text` | string | One document |
| `texts` | list of strings | Several documents; each must be non-empty |
| `thr_score` | float, optional | Minimum cluster membership score; defaults to the server's `thr_score` |
| `use_gpu` | bool, optional | Encode on CUDA; defaults to the server's `use_gpu` |
| `max_length` | int, optional (1–8192) | Encoder chunk length |

The response is `{"entities": [...]}`, one row per linked mention:

| Field | Meaning |
|---|---|
| `mention` | The mention text |
| `a`, `b` | Character offsets in the document |
| `itext` | Index of the document in `texts` |
| `ichunk` | Encoder chunk the mention came from |
| `entity_id_predicted` | The **KB-out id** of the mention's cluster (`<kb>::C0007`). This is the linker's primary output: a cluster may merge several input-KB entries or hold one sense of an entry that is split across clusters |
| `score` | Cluster membership probability; rows below `thr_score` are dropped |
| `direction_predicted` | `forward`, `inverse` or `symmetric`, relative to the canonical relation the cluster represents. Present only when the linker was fitted with a class view (`emits_direction` in `/info`) |

Mentions that fall in no cluster (HDBSCAN noise), or that a screener rejects, are
abstentions: they are omitted rather than linked to the nearest entity.

### From a KB-out id to the input KB

A KB-out id belongs to one fit. To reach the ids of the input KB, read the fit's catalog
(`linker_fit.kb_out.json`, written under the fit's `report_path`):

- `pelinker.kb.kb_out.kb_out_to_kb_in_map(catalog)` gives each KB-out id's dominant
  input-KB id;
- `pelinker.kb.kb_out.kb_out_to_reldir_map(catalog)` gives (dominant input-KB id,
  dominant direction).

The dominant id is a projection. The catalog's `components` and `entity_membership`
record everything a cluster absorbed.

## `POST /link/debug`

The same inputs as `/link`, plus:

| Field | Type | Meaning |
|---|---|---|
| `include_entity_anomaly_metrics` | bool | Add PCA residual, Mahalanobis, spectral entropy and projection scores to each entity row |
| `kb_validation` | bool | Add KB lemma-match fields to each entity row |

The response adds `mention_anomaly`: one diagnostic row per extracted mention, including
the ones a screener rejected.

## Smoke client

`run/smoke_server.py` calls every route against a running server; see
[Run scripts & CLIs](run_scripts_and_cli.md).
