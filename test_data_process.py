from data_process import DataProcess

data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")


def test_plot_data():
    data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")

    data_process.plot_data()


def test_to_three_min():
    data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")

    data_process.to_three_min()


def test_plot_data_distribution():
    data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")

    data_process.plot_data_distribution('pct_change_1')
    data_process.plot_data_distribution('pct_change_2')
    data_process.plot_data_distribution('pct_change_5')


def test_train_model():
    data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")
    data_process.train_model_RF()


def test_get_threshold():
    data_process = DataProcess("LC_20230721_20251030_Adjusted.csv")
    data_process.get_threshold()
