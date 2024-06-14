import pandas as pd


def main():
    ro_df = pd.read_csv("./data/derived/properties.ro.csv")
    go_df = pd.read_csv("./data/derived/properties.go.csv")
    gw_df = pd.read_csv("./data/raw/properties.csv")
    df = pd.concat([ro_df, go_df, gw_df])

    # assign id to ad hoc properties
    df_no_id = df[df["property"].isnull()].sort_values("label").copy()
    df_with_id = df[~df["property"].isnull()].sort_values("label").copy()

    df_no_id = df_no_id.reset_index().drop(["index"], axis=1).reset_index()
    df_no_id["property"] = df_no_id["index"].apply(lambda x: f"GG.{x:04d}")
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
