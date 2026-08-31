import pandas as pd
from pathlib import Path

from pelinker.kb.registry import fetch_latest_kb


def main():
    path_derived = Path("./data/derived/")
    ro_df = pd.read_csv(path_derived / "properties.ro.csv")
    gg_df = pd.read_csv("./data/raw/properties.csv")

    fname, version = fetch_latest_kb(path_derived)
    if fname is not None:
        reference_df = pd.read_csv(path_derived / fname)

        # take only new entities from gg_df : the rest keep from gg_df as they were

        reference_df_pel = reference_df[
            reference_df["entity_id"].apply(lambda x: x.startswith("PEL"))
        ]

        reference_ids = set(reference_df["entity_id"])

        pel_labels_present = reference_df_pel["label"]
        highest_entity_id = sorted(reference_df_pel["entity_id"].tolist())[-1]
        staring_entity_id = int(highest_entity_id.split(".")[1]) + 1

        present_gg_df = reference_df_pel[
            reference_df_pel["label"].isin(pel_labels_present)
        ].copy()

    else:
        pel_labels_present = []
        staring_entity_id = 0
        present_gg_df = pd.DataFrame()
        reference_ids = set()

    add_gg_df = gg_df[~gg_df["label"].isin(pel_labels_present)].copy()

    # process ro ontology
    ro_df = ro_df[ro_df["label"].notnull()].copy()

    patterns = ["not for use in curation", "obsolete"]

    mask = ro_df["label"].apply(lambda x: any(y in x for y in patterns))
    ro_df = ro_df[~mask].copy()
    print(f"number of dropped items in ro_df : {sum(mask)}")

    # RO's `owl:inverseOf` pairs ride along; sources without the column get NA. This is
    # the KB's own statement that two entries are one relation read from two ends — the
    # signal a direction model needs. No `orientation` column is materialized: RO does
    # not name a canonical member of a pair, and inventing one would put an assertion in
    # the KB that no source makes.
    df = pd.concat([ro_df, present_gg_df, add_gg_df])
    if "inverse_entity_id" not in df.columns:
        df["inverse_entity_id"] = pd.NA

    # assign id to ad hoc properties
    df_no_id = df[df["entity_id"].isnull()].sort_values("label").copy()
    df_with_id = df[~df["entity_id"].isnull()].sort_values("label").copy()

    df_no_id["entity_id"] = [
        f"PEL.{ix + staring_entity_id:06d}" for ix in range(df_no_id.shape[0])
    ]

    df = pd.concat([df_with_id, df_no_id])

    # drop property duplicates, keep top item with non-null description
    df = df.sort_values(by=["entity_id", "description"]).drop_duplicates(
        subset=["entity_id"], keep="first"
    )

    # drop label duplicates, keep top item with non-null property
    df = df.sort_values(by=["label", "entity_id"]).drop_duplicates(
        subset=["label"], keep="first"
    )
    print(df.isnull().sum(0))

    # remove null labels
    df = df[~df["label"].isnull()].copy()
    # remove null labels
    df = df[~df["entity_id"].isnull()].copy()
    print(df.isnull().sum(0))
    current_version = version + 1

    # A new version is warranted when the vocabulary changes *or* when the schema does —
    # the id-only check silently swallowed added columns (e.g. inverse_entity_id).
    reference_columns = set(reference_df.columns) if fname is not None else set()
    ids_changed = set(df["entity_id"]) != reference_ids
    columns_changed = set(df.columns) != reference_columns
    if ids_changed or columns_changed:
        out_path = f"./data/derived/properties.synthesis.{current_version}.csv"
        df.sort_values(["entity_id", "label"]).to_csv(out_path, index=False)
        print(
            f"wrote {out_path} (ids changed: {ids_changed}, "
            f"columns changed: {columns_changed}, "
            f"{int(df['inverse_entity_id'].notna().sum())} inverse pairs)"
        )
    else:
        print("no vocabulary or schema change; not writing a new version")


if __name__ == "__main__":
    main()
