import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target, T=24):
    np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
    return 100 * np.mean((np.abs(target - predict) + 1e-5) / (np.abs(target) + 1e-5))


def smape(predict, target, T=24):
    return 200 * np.mean(np.abs(target - predict) / (np.abs(target) + np.abs(predict)))


def mase(predict, target, m=24):
    assert len(target) == len(predict)
    assert m > 1

    period_target = 0
    count = 0
    for i in range(m, len(target)):
        period_target += np.abs(target[i] - target[i-m])
        count += 1
    residuals = np.abs(target - predict)
    np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
    denominator = period_target / (count + 1e-5)

    # 计算MASE
    mase = np.mean(residuals / denominator)

    return mase




