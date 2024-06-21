import logging.config

import click
from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import cross_origin
from flask_restful import Api
from waitress import serve
from importlib.resources import files
import joblib
from pelinker.util import load_models, MAX_LENGTH
from pelinker.model import LinkerModel
import spacy

app = Flask(__name__)
Compress(app)
api = Api(app)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="pubmedbert",
    help="run over BERT flavours",
)
@click.option(
    "--superposition",
    type=click.BOOL,
    default=False,
    help="use a superposition of label and description embeddings, where available",
)
@click.option(
    "--layers",
    type=click.INT,
    default=[-6, -5, -4, -3, -2, -1],
    multiple=True,
    help="layers to consider",
)
@click.option("--port", type=click.INT, default=8599)
@click.option("--host", type=click.STRING, default="0.0.0.0")
def main(model_type, layers, superposition, port, host):
    extra_context = False
    suffix = ".superposition" if superposition else ""
    layers_str = LinkerModel.encode_layers(layers)

    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    logger.debug("debug is on")
    app.logger.setLevel(logging.INFO)

    file_path = files("pelinker.store").joinpath(
        f"pelinker.model.{model_type}.{layers_str}{suffix}.gz"
    )

    pe_model: LinkerModel = joblib.load(file_path)

    tokenizer, model = load_models(model_type)

    nlp = spacy.load("en_core_web_sm")

    @app.route("/pelinker", methods=["POST"])
    @cross_origin()
    def link():
        """
            navigate api
                1. converts incoming text to a graph
                2. computes metrics of the resultant graph wrt to the literature KG
                3. fetches the graphs of most relevant publications from the literature KG

        :return: response
        """
        if request.method == "POST":
            json_data = request.json
            try:
                text = json_data["text"]
                r = pe_model.link(
                    text, tokenizer, model, nlp, MAX_LENGTH, extra_context
                )

            except Exception as exc:
                return {"error": str(exc)}, 202

            try:
                json_response = jsonify(r)
            except Exception as exc:
                return {"error": str(exc)}, 201

            return json_response, 200

    serve(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
