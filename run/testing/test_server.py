from pprint import pprint
import requests

if __name__ == "__main__":
    text = "Diabetic ulcers are related to burns."
    ip = "localhost"
    url = f"http://{ip}:8599/pelinker"
    r = requests.post(url, json={"text": text}, verify=False).json()
    pprint(r)
