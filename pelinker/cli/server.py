from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

import hydra
import spacy
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from pelinker.config import EmbeddingModelMetadata, KBConfig
from pelinker.model import Linker
from pelinker.onto import MAX_LENGTH
from pelinker.util import str2layers, layers2str

logger = logging.getLogger(__name__)


@dataclass
class ServerCliConfig:
    """Hydra/OmegaConf node for the linker HTTP server (aligned with ``fit`` CLI)."""

    host: str = "0.0.0.0"
    port: int = 8599
    """Path to the dumped linker **without** ``.gz`` (same as ``Linker.dump`` / ``Linker.load``). If omitted, uses the packaged store path built from ``model_type`` and ``layers_spec``."""
    model_file_spec: str | None = None
    model_type: str = "pubmedbert"
    layers_spec: str = "1"
    thr_score: float = 0.5
    nlp_model: str = "en_core_web_trf"
    use_gpu: bool = False
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    thr_score: float | None = None
    use_gpu: bool | None = None


class ServerState:
    def __init__(
        self,
        *,
        linker: Linker,
        nlp: Any,
        cfg: ServerCliConfig,
        resolved_model_path: str,
    ) -> None:
        self.linker = linker
        self.nlp = nlp
        self.cfg = cfg
        self.resolved_model_path = resolved_model_path


def _resolve_model_file_spec(cfg: ServerCliConfig) -> Any:
    if cfg.model_file_spec:
        p = Path(cfg.model_file_spec).expanduser()
        if p.suffix == ".gz":
            p = p.with_suffix("")
        return p
    layers = str2layers(cfg.layers_spec)
    layers_str = layers2str(layers)
    return files("pelinker.store").joinpath(
        f"pelinker.model.{cfg.model_type}.{layers_str}"
    )


def _load_linker(cfg: ServerCliConfig) -> tuple[Linker, str]:
    file_spec = _resolve_model_file_spec(cfg)
    resolved = str(file_spec)
    try:
        linker = Linker.load(file_spec)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No dumped model at {resolved!s}.gz (check model_file_spec or model_type/layers_spec)."
        ) from exc

    if linker.embedding_metadata is None:
        linker.embedding_metadata = EmbeddingModelMetadata.from_single(
            cfg.model_type, cfg.layers_spec
        )
        logger.warning(
            "Loaded linker had no embedding_metadata; using cfg: model_type=%r layers_spec=%r",
            cfg.model_type,
            cfg.layers_spec,
        )
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


def build_info_payload(state: ServerState) -> dict[str, Any]:
    linker = state.linker
    em = linker.embedding_metadata
    sources_json: list[dict[str, str]] = []
    if em is not None:
        sources_json = [
            {"model_type": s.model_type, "layers_spec": s.layers_spec}
            for s in em.sources
        ]
    cluster_ids = set(linker.cluster_assignments.values())
    return {
        "resolved_model_path": state.resolved_model_path,
        "embedding_metadata": {"sources": sources_json} if em is not None else None,
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
        logger.info("Loading spaCy: %s", cfg.nlp_model)
        nlp = spacy.load(cfg.nlp_model)
        state_holder["state"] = ServerState(
            linker=linker,
            nlp=nlp,
            cfg=cfg,
            resolved_model_path=resolved,
        )
        yield
        state_holder["state"] = None

    app = FastAPI(title="pelinker", lifespan=lifespan)
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

    StateDep = Annotated[ServerState, Depends(get_state)]

    @app.get("/info")
    def info(state: StateDep) -> dict[str, Any]:
        return build_info_payload(state)

    @app.post("/link")
    def link(body: LinkRequest, state: StateDep) -> dict[str, Any]:
        thr_s = body.thr_score if body.thr_score is not None else state.cfg.thr_score
        use_gpu = body.use_gpu if body.use_gpu is not None else state.cfg.use_gpu
        try:
            r = state.linker.predict(
                [body.text],
                state.nlp,
                MAX_LENGTH,
                threshold=0.0,
                use_gpu=use_gpu,
            )
            r = Linker.filter_report(r, thr_score=thr_s)
        except Exception as exc:
            logger.exception("link failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return r

    return app


def serve(cfg: ServerCliConfig) -> None:
    app = create_app(cfg)
    logger.info("Starting uvicorn on %s:%s", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="server_config", node=ServerCliConfig)


@hydra.main(version_base=None, config_path=None, config_name="server_config")
def run(cfg: ServerCliConfig) -> None:
    logger.info("Running server with config:\n%s", OmegaConf.to_yaml(cfg))
    try:
        serve(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
