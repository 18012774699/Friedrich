import numpy as np


# 这个函数可以根据要求创建出时间序列（通过batch_size参数），长度为n_steps，每个时间步只有1个值。
# 函数返回NumPy数组，形状是[批次大小, 时间步数, 1]，每个序列是两个正弦波之和（固定强度+随机频率和相位），加一点噪音。
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))    # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))   # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)     # + noise
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
