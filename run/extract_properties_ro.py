import pandas as pd
import rdflib
from rdflib import Namespace
from rdflib.namespace import OWL, RDFS


def main():
    g = rdflib.Graph()
    g.parse("./data/raw/ro.owl", format="xml")

    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("obo", Namespace("http://purl.obolibrary.org/obo"))

    query = """
    SELECT ?subject
    WHERE {
        ?subject rdf:type owl:ObjectProperty .
    }
    """
    r = g.query(query)
    properties = [x.subject for x in r]

    query = """
    SELECT ?subject ?label
    WHERE {
        ?subject rdfs:label ?label .
    }
    """
    r = g.query(query)
    labels = [x for x in r]

    query = """
    SELECT ?subject ?d
    WHERE {
        ?subject obo:IAO_0000115 ?d .
    }
    """
    r = g.query(query)

    descs = [x for x in r]

    ids = [x.split("/")[-1] for x in properties]
    ids = [".".join(x.split("_")) for x in ids]

    props_df = pd.DataFrame(list(zip(properties, ids)), columns=["iri", "property"])
    labels_df = pd.DataFrame(labels, columns=["iri", "label"])
    desc_df = pd.DataFrame(descs, columns=["iri", "description"])

    ro_df = props_df.merge(labels_df, how="left", on="iri").merge(
        desc_df, how="left", on="iri"
    )

    ro_df = ro_df.drop("iri", axis=1)

    ro_df.to_csv("./data/derived/properties.ro.csv", index=False)


if __name__ == "__main__":
    main()
