import pandas as pd


def main():
    ro_df = pd.read_csv("./data/derived/properties.ro.csv")
    gg_df = pd.read_csv("./data/raw/properties.csv")

    ro_df = ro_df[ro_df["label"].notnull()].copy()

    patterns = ["not for use in curation", "obsolete"]

    mask = ro_df["label"].apply(lambda x: any(y in x for y in patterns))
    ro_df = ro_df[~mask].copy()
    print(f"number of dropped items in ro_df : {sum(mask)}")

    df = pd.concat([ro_df, gg_df])

    # assign id to ad hoc properties
    df_no_id = df[df["property"].isnull()].sort_values("label").copy()
    df_with_id = df[~df["property"].isnull()].sort_values("label").copy()

    df_no_id = df_no_id.reset_index().drop(["index"], axis=1).reset_index()
    df_no_id["property"] = df_no_id["index"].apply(lambda x: f"PEL.{x:06d}")
    df_no_id = df_no_id.drop(["index"], axis=1)

    df = pd.concat([df_with_id, df_no_id])

    # drop property duplicates, keep top item with non null description
    df = df.sort_values(by=["property", "description"]).drop_duplicates(
        subset=["property"], keep="first"
    )

    # drop label duplicates, keep top item with non null property
    df = df.sort_values(by=["label", "property"]).drop_duplicates(
        subset=["label"], keep="first"
    )
    print(df.isnull().sum(0))

    # remove null labels
    df = df[~df["label"].isnull()].copy()
    # remove null labels
    df = df[~df["property"].isnull()].copy()
    print(df.isnull().sum(0))
    df.sort_values(["property", "label"]).to_csv(
        "./data/derived/properties.synthesis.csv", index=False
    )


if __name__ == "__main__":
    main()
