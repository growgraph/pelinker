import gzip
import time

import numpy as np
import pandas as pd
import rdflib
import requests


def fetch_url(ent_id):
    url = (
        f"https://www.ebi.ac.uk/ols4/api/ontologies/ro/properties"
        f"/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F{ent_id}"
    )
    response = requests.get(url)
    code = response.status_code
    if code == 200:
        json_data = response.json()
        return code, json_data
    else:
        print(f"Failed to retrieve entity: {ent_id}")
        return code, ent_id


def main():
    g = rdflib.Graph()
    with gzip.open("./data/raw/GO-CAMs.ttl.gz", "rt", encoding="utf-8") as f:
        g.parse(f, format="ttl")
    print(len(g))

    query = """
    SELECT ?subject
    WHERE {
        ?subject rdf:type owl:ObjectProperty .
    }
    """
    r = g.query(query)
    relations = sorted([x.subject for x in r])

    go_data = []
    errors = []
    for row in relations:
        ent_id = row.split("/")[-1]
        code, response = fetch_url(ent_id)
        if code == 200:
            go_data += [response]
        else:
            errors += [response]
        sleep_sec = np.random.uniform(0.0001, 0.2)
        time.sleep(sleep_sec)

    ids = [jd["iri"].split("/")[-1] for jd in go_data]
    ids = [".".join(x.split("_")) for x in ids]
    go_data2 = [
        (_id, jd["label"], jd["description"][0] if jd["description"] else np.nan)
        for _id, jd in zip(ids, go_data)
    ]

    df_go = pd.DataFrame(go_data2, columns=["entity_id", "label", "description"])

    pd.Series(errors).to_csv("./data/derived/properties.go.failed.csv")
    df_go.to_csv("./data/derived/properties.go.csv", index=False)


if __name__ == "__main__":
    main()
