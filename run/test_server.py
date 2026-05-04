"""HTTP smoke tests against ``pelinker.cli.server`` (run the server separately)."""

from __future__ import annotations

import json
import pathlib
import sys
from pprint import pprint
from typing import Any

import click
import requests

from pelinker.io import load_json_path

_DEFAULT_LINK_BODY: dict[str, Any] = {
    "texts": [
        (
            "Rainfall causes the river level to rise rapidly. The elevated water "
            "level constrains road access in nearby villages. Emergency response "
            "operations occur in the flooded area."
        ),
        (
            "The policy framework includes several regulatory mechanisms as parts. "
            "These mechanisms govern industrial emissions. Their enforcement "
            "motivates firms to adopt cleaner technologies."
        ),
    ]
}


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}".rstrip("/")


def _resolve_url(base: str, path: str) -> str:
    return f"{base}{path}" if path.startswith("/") else f"{base}/{path}"


def _load_link_body(path: pathlib.Path) -> dict[str, Any]:
    raw = load_json_path(path)
    if not isinstance(raw, dict):
        raise click.ClickException(
            f"JSON root must be an object, got {type(raw).__name__}"
        )
    if raw.get("text") is None and raw.get("texts") is None:
        raise click.ClickException(
            "JSON must include 'text' or 'texts' (see LinkRequest in server)."
        )
    return raw


def _emit_result(data: Any, output: pathlib.Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        click.echo(f"Wrote {output}", err=True)
    else:
        pprint(data, stream=sys.stdout)


def _request_json(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    if method == "GET":
        response = requests.get(url, timeout=timeout)
    else:
        response = requests.post(url, json=json_body, timeout=timeout)
    if not response.ok:
        detail = response.text.strip() or "(empty body)"
        raise click.ClickException(
            f"{response.status_code} {response.reason} — {detail}"
        )
    if not response.content.strip():
        return None
    ctype = response.headers.get("content-type", "")
    if "application/json" in ctype:
        return response.json()
    return response.text


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--host", type=str, default="localhost", show_default=True)
@click.option(
    "--port",
    type=int,
    default=8599,
    show_default=True,
    help="Must match the running server (see pelinker.cli.server ServerCliConfig).",
)
@click.option(
    "--endpoint",
    type=click.Choice(["health", "info", "model", "link", "link-debug"]),
    default="link",
    show_default=True,
    help="Which route to call (GET for health/info/model, POST for link).",
)
@click.option(
    "--input-path",
    type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False, readable=True),
    help="JSON body for POST endpoints; must include 'text' or 'texts' (optional thr_score, use_gpu, …).",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Write JSON response to this path instead of printing.",
)
@click.option("--timeout", type=float, default=300.0, show_default=True)
def main(
    host: str,
    port: int,
    endpoint: str,
    input_path: pathlib.Path | None,
    output: pathlib.Path | None,
    timeout: float,
) -> None:
    """Call a pelinker HTTP server (``uv run python -m pelinker.cli.server``)."""
    base = _base_url(host, port)
    if endpoint == "health":
        url = _resolve_url(base, "/health")
        data = _request_json("GET", url, timeout=timeout)
    elif endpoint == "info":
        url = _resolve_url(base, "/info")
        data = _request_json("GET", url, timeout=timeout)
    elif endpoint == "model":
        url = _resolve_url(base, "/model")
        data = _request_json("GET", url, timeout=timeout)
    elif endpoint == "link":
        body = _load_link_body(input_path) if input_path else dict(_DEFAULT_LINK_BODY)
        url = _resolve_url(base, "/link")
        data = _request_json("POST", url, json_body=body, timeout=timeout)
    else:
        body = _load_link_body(input_path) if input_path else dict(_DEFAULT_LINK_BODY)
        url = _resolve_url(base, "/link/debug")
        data = _request_json("POST", url, json_body=body, timeout=timeout)

    _emit_result(data, output)


if __name__ == "__main__":
    main()
