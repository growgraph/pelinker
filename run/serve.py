import logging.config

import click
from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import cross_origin
from flask_restful import Api
from waitress import serve
from importlib.resources import files
from pelinker.util import load_models, MAX_LENGTH
from pelinker.model import LinkerModel
import spacy

app = Flask(__name__)
Compress(app)
api = Api(app)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", type=click.INT, default=8599)
@click.option("--host", type=click.STRING, default="0.0.0.0")
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
@click.option("--extra-context", type=click.BOOL, is_flag=True, default=False)
@click.option(
    "--layers-spec",
    default="sent",
    type=click.STRING,
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
@click.option("--thr-score", type=click.FLOAT, default=0.5)
@click.option("--thr-dif", type=click.FLOAT, default=0.0)
def main(
    model_type,
    layers_spec,
    superposition,
    port,
    host,
    extra_context,
    thr_score,
    thr_dif,
):
    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    app.logger.setLevel(logging.INFO)

    logger.info(f"thr_score : {thr_score}, thr_dif: {thr_dif}")

    layers = LinkerModel.str2layers(layers_spec)
    sentence = True if layers == "sent" else False
    suffix = ".superposition" if superposition else ""
    layers_str = LinkerModel.layers2str(layers)

    model_spec = files("pelinker.store").joinpath(
        f"pelinker.model.{model_type}.{layers_str}{suffix}"
    )

    logger.info(f"model path: {model_spec}")
    pe_model = LinkerModel.load(model_spec)

    logger.info(f"pelinker model loaded : {pe_model}")

    tokenizer, model = load_models(model_type, sentence=sentence)
    logger.info(f"tokenizer and model loaded : {model}")

    nlp = spacy.load("en_core_web_trf")
    logger.info(f"spacy model loaded : {nlp}")

    @app.route("/link", methods=["POST"])
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
                thr_score0 = json_data.pop("thr_score", thr_score)
                thr_dif0 = json_data.pop("thr_score", thr_dif)
                r = pe_model.link(
                    text, tokenizer, model, nlp, MAX_LENGTH, extra_context
                )
                r = LinkerModel.filter_report(r, thr_score=thr_score0, thr_dif=thr_dif0)

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
