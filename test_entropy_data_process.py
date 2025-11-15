from entropy_data_process import EntropyDataProcess


def test_calc_entropy_avg():
    entropy = EntropyDataProcess('LC_20230721_20251030_Adjusted.csv')
    res = entropy.calc_entropy_avg_as_threshold()
    print(res)


def test_resample_to_entropy():
    entropy = EntropyDataProcess('LC_20230721_20251030_Adjusted.csv')
    res = entropy.resample_to_entropy()


def test_get_pct():
    entropy = EntropyDataProcess('LC_20230721_20251030_Adjusted.csv')
    entropy.get_threshold()


def test_train_model_rf():
    entropy = EntropyDataProcess('LC_20230721_20251030_Adjusted.csv')
    entropy.train_model_RF()
