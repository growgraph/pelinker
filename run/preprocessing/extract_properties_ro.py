import pandas as pd
import rdflib
from rdflib import Namespace
from rdflib.namespace import OWL, RDFS


def iri_to_entity_id(iri: str) -> str:
    """``.../RO_0002206`` → ``RO.0002206`` (the id form used across the KB)."""
    return ".".join(str(iri).split("/")[-1].split("_"))


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

    # Inverse pairs are the KB's own statement of predicate orientation: "activates" /
    # "activated by" are one relation read from two ends. They were dropped here before,
    # which left the pipeline with no way to tell a predicate from its converse.
    query = """
    SELECT ?subject ?inverse
    WHERE {
        ?subject owl:inverseOf ?inverse .
    }
    """
    inverses = [x for x in g.query(query)]

    ids = [iri_to_entity_id(x) for x in properties]

    props_df = pd.DataFrame(list(zip(properties, ids)), columns=["iri", "entity_id"])
    labels_df = pd.DataFrame(labels, columns=["iri", "label"])
    desc_df = pd.DataFrame(descs, columns=["iri", "description"])

    ro_df = props_df.merge(labels_df, how="left", on="iri").merge(
        desc_df, how="left", on="iri"
    )

    inverse_df = pd.DataFrame(inverses, columns=["iri", "inverse_iri"])
    if not inverse_df.empty:
        # Only keep blank-node-free, resolvable pairs; RO states some inverses against
        # anonymous property expressions, which have no entity id to point at.
        inverse_df = inverse_df.loc[
            inverse_df["inverse_iri"].map(lambda x: isinstance(x, rdflib.URIRef))
        ].copy()
        inverse_df["inverse_entity_id"] = inverse_df["inverse_iri"].map(
            iri_to_entity_id
        )
        inverse_df = inverse_df.drop("inverse_iri", axis=1).drop_duplicates("iri")
        ro_df = ro_df.merge(inverse_df, how="left", on="iri")
    else:
        ro_df["inverse_entity_id"] = pd.NA

    ro_df = ro_df.drop("iri", axis=1)

    n_pairs = int(ro_df["inverse_entity_id"].notna().sum())
    print(f"Extracted {len(ro_df)} properties, {n_pairs} with a declared inverse")

    ro_df.to_csv("./data/derived/properties.ro.csv", index=False)


if __name__ == "__main__":
    main()
