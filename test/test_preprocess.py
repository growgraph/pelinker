from pelinker.preprocess import pre_process_properties


def test_prep(df_properties):
    pre_process_properties(df_properties)
    assert True
