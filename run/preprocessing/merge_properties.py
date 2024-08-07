import pandas as pd
from pathlib import Path


def fetch_latest_kb(path_derived) -> tuple[str | None, int]:
    file_names = [
        file.name
        for file in path_derived.iterdir()
        if file.is_file() and ".synthesis." in file.name
    ]
    filename_versions = sorted(
        [(f, int(f.split(".")[-2])) for f in file_names], key=lambda x: x[1]
    )
    if filename_versions:
        return filename_versions[-1]
    else:
        return None, -1


# TODO merge should respect the following:
# if there is a previous version of properties.synthesis.0.csv


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

    df = pd.concat([ro_df, present_gg_df, add_gg_df])

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

    if set(df["entity_id"]) != reference_ids:
        df.sort_values(["entity_id", "label"]).to_csv(
            f"./data/derived/properties.synthesis.{current_version}.csv", index=False
        )


if __name__ == "__main__":
    main()
