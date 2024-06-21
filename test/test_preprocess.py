from pelinker.preprocess import pre_process_properties


def test_prep(df_properties):
    report = pre_process_properties(df_properties)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    # ixlabel_ixdesc = report.pop("ixlabel_ixdesc")
    # properties = report.pop("properties")
    # property_label_map = report.pop("property_label_map")
    assert len(labels) == 685
    assert len(descriptions) == 526
