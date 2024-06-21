from pprint import pprint
import requests

if __name__ == "__main__":
    text = "Diabetic ulcers are related to burns."
    ip = "192.168.1.1"
    ip = "127.0.0.1"
    url = f"http://{ip}:8599/pelinker"
    r = requests.post(url, json={"text": text}, verify=False).json()
    pprint(r)
