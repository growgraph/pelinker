from pprint import pprint
import requests
import click


@click.command()
@click.option("--port", type=click.INT, default=8599)
@click.option("--host", type=click.STRING, default="localhost")
def run(host, port):
    text = (
        "TAMs can also secrete in the TME a number of immunosuppressive cytokines, "
        "such as IL-6, TGF-β, and IL-10 that are able to suppress CD8+ T-cell function."
    )
    text = "Objectives"

    url = f"http://{host}:{port}/link"
    r = requests.post(url, json={"text": text}, verify=False)
    jr = r.json()
    pprint(jr)


if __name__ == "__main__":
    run()
