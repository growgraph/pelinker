from pprint import pprint
import requests

if __name__ == "__main__":
    text = (
        "Specifically, IL-6 is expressed at high levels in PDAC, "
        "and its increasing circulating level is associated with advanced disease and poor prognosis (77)."
    )
    ip = "localhost"
    url = f"http://{ip}:8599/pelinker"
    r = requests.post(url, json={"text": text}, verify=False).json()
    pprint(r)
