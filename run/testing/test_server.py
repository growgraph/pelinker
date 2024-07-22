from pprint import pprint
import requests
import click


@click.command()
@click.option("--port", type=click.INT, default=8599)
@click.option("--host", type=click.STRING, default="localhost")
def run(host, port):
    text = (
        "Specifically, IL-6 is expressed at high levels in PDAC, "
        "and its increasing circulating level is associated with advanced disease and poor prognosis (77)."
    )
    url = f"http://{host}:{port}/link"
    r = requests.post(url, json={"text": text}, verify=False)
    jr = r.json()
    pprint(jr)


if __name__ == "__main__":
    run()
