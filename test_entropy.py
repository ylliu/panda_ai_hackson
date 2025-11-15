import numpy as np

from entropy import SampleEntropy


def test_sample_entropy():
    from entropy import SampleEntropy
    import pandas as pd
    import numpy as np

    # 自定义序列
    data = np.array([0.1, 0.1, 0.1, 0.5])
    se = SampleEntropy()
    H = se.ksg_entropy_1d(data)
    print("KSG entropy (1D) =", H)


def test_sample_entropy_pyentrp():
    values = [0.1, 0.1, 0.1, 0.1]
    se = SampleEntropy()
    res = se.sample_entropy_pyentrp(values)

    print(res[0])


def test_ksg_entropy():
    se = SampleEntropy()
    se.ksg_entropy([0.1, 0.3, 0.2, -0.1])


def test_renyi_entropy():
    se = SampleEntropy()
    x = np.array([0.1, 0.1, 0.1, 0.1])
    H = se.renyi_entropy(x, sigma=2, alpha=0.01)
    print("Renyi entropy =", H)
