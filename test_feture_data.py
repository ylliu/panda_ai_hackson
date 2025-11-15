import pandas as pd

from feture_data import FutureData


def test_test_panda_ai():
    df = FutureData().test_panda_ai()


def test_get_master_future():
    df = FutureData().get_master_future()
    print(df)


def test_adjust_back_adjustment():
    future = FutureData()
    df = pd.read_csv("LC_20230721_20251030.csv")
    future.adjust_back_adjustment(df)
