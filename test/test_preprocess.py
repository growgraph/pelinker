from pelinker.preprocess import pre_process_properties


def test_prep(df_properties):
    report = pre_process_properties(df_properties)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    assert len(labels) == 689
    assert len(descriptions) == 529
