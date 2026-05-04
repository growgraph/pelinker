from __future__ import annotations
from fastapi.staticfiles import StaticFiles
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

from typing_extensions import Self
from fastapi.openapi.docs import get_swagger_ui_html

import hydra
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.gzip import GZipMiddleware

from pelinker.config import EmbeddingModelMetadata, KBConfig
from pelinker.model import Linker
from pelinker.onto import MAX_LENGTH

logger = logging.getLogger(__name__)

# Repo-root ``static/`` (sibling of the ``pelinker`` package), not CWD-relative.
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


@dataclass
class ServerCliConfig:
    """Hydra/OmegaConf node for the linker HTTP server (aligned with ``fit`` CLI)."""

    host: str = "0.0.0.0"
    port: int = 8599
    """Path to the dumped linker **without** ``.gz`` (same as ``Linker.dump`` / ``Linker.load``). If omitted, uses the packaged store path built from ``model_type`` and ``layers_spec``."""
    model_file_spec: str | None = None
    thr_score: float = 0.5
    use_gpu: bool = False
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])


class _LinkTextsBody(BaseModel):
    """Shared ``text`` / ``texts`` input shape for ``/link`` and ``/link/debug``."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = Field(default=None, min_length=1)
    texts: list[str] | None = None

    @model_validator(mode="after")
    def _coalesce_texts(self) -> Self:
        if self.texts is not None:
            if not self.texts:
                raise ValueError("texts must contain at least one item")
            for i, t in enumerate(self.texts):
                if not str(t).strip():
                    raise ValueError(f"texts[{i}] must be non-empty")
            return self
        if self.text is not None:
            object.__setattr__(self, "texts", [self.text])
            return self
        raise ValueError("Provide 'text' or non-empty 'texts'")


class LinkRequest(_LinkTextsBody):
    thr_score: float | None = None
    use_gpu: bool | None = None
    max_length: int | None = Field(default=None, ge=1, le=8192)


class LinkDebugRequest(_LinkTextsBody):
    """Same inputs as ``/link`` plus flags matching ``pelinker.cli.link_files``."""

    thr_score: float | None = None
    use_gpu: bool | None = None
    max_length: int | None = Field(default=None, ge=1, le=8192)
    include_entity_anomaly_metrics: bool = False
    kb_validation: bool = False


class ServerState:
    def __init__(
        self,
        *,
        linker: Linker,
        cfg: ServerCliConfig,
        resolved_model_path: str,
    ) -> None:
        self.linker = linker
        self.cfg = cfg
        self.resolved_model_path = resolved_model_path


def _resolve_model_file_spec(cfg: ServerCliConfig) -> Any:
    if cfg.model_file_spec:
        p = Path(cfg.model_file_spec).expanduser()
        if p.suffix == ".gz":
            p = p.with_suffix("")
        return p
    return files("pelinker.store").joinpath("model.v0")


def _load_linker(cfg: ServerCliConfig) -> tuple[Linker, str]:
    file_spec = _resolve_model_file_spec(cfg)
    resolved = str(file_spec)
    try:
        linker = Linker.load(file_spec)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No dumped model at {resolved!s}.gz (check model_file_spec or model_type/layers_spec)."
        ) from exc

    return linker, resolved


def _kb_to_jsonable(kb: KBConfig | None) -> dict[str, Any] | None:
    if kb is None:
        return None
    d = asdict(kb)
    created = d["created_at"]
    if isinstance(created, date):
        d["created_at"] = created.isoformat()
    return d


def _transform_to_jsonable(linker: Linker) -> dict[str, Any] | None:
    tc = linker.transform_config
    if tc is None:
        return None
    return asdict(tc)


def _embedding_metadata_to_json(
    em: EmbeddingModelMetadata | None,
) -> dict[str, Any] | None:
    """Full serialization of :class:`~pelinker.config.EmbeddingModelMetadata` from the artifact."""
    if em is None:
        return None
    return asdict(em)


def build_info_payload(state: ServerState) -> dict[str, Any]:
    linker = state.linker
    em = linker.embedding_metadata
    cluster_ids = set(linker.cluster_assignments.values())
    return {
        "resolved_model_path": state.resolved_model_path,
        "embedding_metadata": _embedding_metadata_to_json(em),
        "nlp_model_name": linker.nlp_model_name,
        "kb": _kb_to_jsonable(linker.kb_config),
        "vocabulary_size": len(linker.vocabulary),
        "cluster_count": len(cluster_ids),
        "transform_config": _transform_to_jsonable(linker),
    }


def create_app(cfg: ServerCliConfig) -> FastAPI:
    state_holder: dict[str, ServerState | None] = {"state": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        linker, resolved = _load_linker(cfg)
        logger.info("Loaded linker from %s.gz", resolved)
        state_holder["state"] = ServerState(
            linker=linker,
            cfg=cfg,
            resolved_model_path=resolved,
        )
        yield
        state_holder["state"] = None

    app = FastAPI(title="pelinker", lifespan=lifespan, docs_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_state() -> ServerState:
        s = state_holder["state"]
        if s is None:
            raise HTTPException(status_code=503, detail="Server not ready")
        return s

    if _STATIC_DIR.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

    _favicon_url = (
        "/static/favicon.ico"
        if (_STATIC_DIR / "favicon.ico").is_file()
        else "https://fastapi.tiangolo.com/img/favicon.png"
    )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="Pelinker API Docs",
            swagger_favicon_url=_favicon_url,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info(state: ServerState = Depends(get_state)) -> dict[str, Any]:
        return build_info_payload(state)

    @app.get("/model")
    def model_info(state: ServerState = Depends(get_state)) -> dict[str, Any]:
        """Embedding identity and spaCy pipeline name (from the loaded linker artifact)."""
        linker = state.linker
        return {
            "embedding_metadata": _embedding_metadata_to_json(
                linker.embedding_metadata
            ),
            "nlp_model_name": linker.nlp_model_name,
        }

    @app.post("/link")
    def link(
        body: LinkRequest, state: ServerState = Depends(get_state)
    ) -> dict[str, Any]:
        assert body.texts is not None
        thr_s = body.thr_score if body.thr_score is not None else state.cfg.thr_score
        use_gpu = body.use_gpu if body.use_gpu is not None else state.cfg.use_gpu
        max_len = body.max_length if body.max_length is not None else MAX_LENGTH
        try:
            pres = state.linker.predict(
                body.texts,
                max_length=max_len,
                threshold=0.0,
                use_gpu=use_gpu,
            )
            r = pres.filter_by_score(thr_s).to_dict(public_entity_fields=True)
        except Exception as exc:
            logger.exception("link failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return r

    @app.post("/link/debug")
    def link_debug(
        body: LinkDebugRequest, state: ServerState = Depends(get_state)
    ) -> dict[str, Any]:
        """Predict with per-mention diagnostics (``mention_anomaly``) and optional KB validation."""
        assert body.texts is not None
        thr_s = body.thr_score if body.thr_score is not None else state.cfg.thr_score
        use_gpu = body.use_gpu if body.use_gpu is not None else state.cfg.use_gpu
        max_len = body.max_length if body.max_length is not None else MAX_LENGTH
        try:
            pres = state.linker.predict(
                body.texts,
                max_length=max_len,
                threshold=0.0,
                use_gpu=use_gpu,
                include_mention_anomaly=True,
                include_prediction_kb_validation=body.kb_validation,
            )
            filtered = pres.filter_by_score(thr_s)
            return filtered.to_dict(
                include_entity_anomaly_metrics=body.include_entity_anomaly_metrics,
                include_debug=True,
            )
        except Exception as exc:
            logger.exception("link/debug failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def serve(cfg: ServerCliConfig) -> None:
    app = create_app(cfg)
    logger.info("Starting uvicorn on %s:%s", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="server_config", node=ServerCliConfig)


@hydra.main(version_base=None, config_path="pkg://pelinker.conf", config_name="server")
def run(cfg: ServerCliConfig) -> None:
    logger.info("Running server with config:\n%s", OmegaConf.to_yaml(cfg))
    try:
        serve(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
