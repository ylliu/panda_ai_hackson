from signal_conversion import SignalConversion


def test_fftsmooth():
    signal_convert = SignalConversion('LC_20230721_20251030_Adjusted.csv')
    signal_convert.FFTSmooth()


def test_wavelet_smooth():
    signal_convert = SignalConversion('LC_20230721_20251030_Adjusted.csv')
    signal_convert.WaveletSmooth()


def test_kalman_smooth():
    signal_convert = SignalConversion('LC_20230721_20251030_Adjusted.csv')
    signal_convert.KalmanSmooth()


def test_zlow_pass_filter():
    signal_convert = SignalConversion('LC_20230721_20251030_Adjusted.csv')
    signal_convert.ZLowPassFilter(alpha=0.5)
