from volume_data_process import VolumeDataProcess


def test_to_min_of():
    volume_data = VolumeDataProcess("LC_20230721_20251030_Adjusted.csv")
    minutes = [4]
    for min in minutes:
        volume_data.to_min_of(min)


def test_plot_volume_over_time():
    volume_data = VolumeDataProcess("LC_20230721_20251030_Adjusted.csv")
    minutes = [4]
    for min in minutes:
        volume_data.plot_volume_over_time(min)


def test_plot_volume_bars():
    volume_data = VolumeDataProcess("LC_20230721_20251030_Adjusted.csv")
    df = volume_data.resample_to_volume(volume_threshold=4780)
    # volume_data.plot_volume_bars(df, title='Volume Bars with Threshold 5800')


def test_get_threshold():
    volume_data = VolumeDataProcess("LC_20230721_20251030_Adjusted.csv")
    volume_data.get_threshold()


def test_train_model_rf():
    volume_data = VolumeDataProcess("LC_20230721_20251030_Adjusted.csv")
    volume_data.train_model_RF()
