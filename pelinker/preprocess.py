def pre_process_properties(df):
    df = df.copy()

    df = df.sort_values("property")

    mask_desc_exclude = df["description"].isnull() | df["description"].apply(
        lambda x: "inverse" in x.lower() if isinstance(x, str) else True
    )

    df_desc = df.loc[mask_desc_exclude].copy()

    # prop -> label
    # id_label_dict = dict(df[["property", "label"]].values)

    # idesc -> description
    # id_description_dict = dict(
    #     df_desc[["property", "description"]].values
    # )

    # (il) : property
    property_from_label = df["property"].reset_index(drop=True).tolist()
    labels = df["label"].values.tolist()
    properties = df["property"].values.tolist()

    # il -> property
    ixlabel_id_dict = dict(enumerate(property_from_label))

    # property -> il
    id_ixlabel_map = {v: k for k, v in ixlabel_id_dict.items()}

    # (ip) : property
    property_from_desc = df_desc["property"].reset_index(drop=True).tolist()
    descriptions = df_desc["description"].values.tolist()

    # il -> ip
    ixlabel_ixdesc = {
        id_ixlabel_map[prop]: ixd for ixd, prop in enumerate(property_from_desc)
    }

    # ip -> property
    # df_desc["property"].reset_index(drop=True)

    report = {
        "labels": labels,
        "descriptions": descriptions,
        "ixlabel_ixdesc": ixlabel_ixdesc,
        "property_from_desc": property_from_desc,
        "id_ixlabel_map": id_ixlabel_map,
        "properties": properties,
    }
    return report
