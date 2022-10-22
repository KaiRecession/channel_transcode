import math
import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from data_load_fix import data_write

seed = 699
power = 120

noise = math.pow(10, -17.4) * 0.001
fc = 15000
# 设置随机种子
rng = np.random.RandomState(seed)
tf.random.set_seed(seed)
# 子载波数量和带宽MBPS的对应关系：
# [0.9, 1.8, 2.6, 3.5, 4.3, 5.1, 6.0, 6.8, 7.7, 8.5]
N_choice = [[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [5, 6, 7]]
r_pro = [[0., 0., 0., 1.],
         [0., 0., 1., 0.],
         [0., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0.25, 0.25, 0.25, 0.25],
         [0.6, 0.2, 0.2, 0.1],
         [0.2, 0.6, 0.2, 0.1],
         [0.1, 0.1, 0.2, 0.6],
         [0.1, 0.1, 0.6, 0.2],
         ]
c_pro = [[0.2, 0.6, 0.2],
         [0.6, 0.2, 0.2],
         [0.2, 0.2, 0.6]]
# 这两个参数可以改变网络的波动频率
r_change = 10  # 大波动时间
c_change = 1  # 小波动时间


# 时间刻度定为100ms，time_length的单位为s
def generate_CSI(time_length):
    num = time_length * 10
    CH = 1 / np.sqrt(2) * (rng.randn(1, num) + 1j * rng.randn(1, num))
    CH = abs(CH)
    CH = np.squeeze(CH)
    return CH


def calc_rate(CH, r_prob, c_prob):
    # CH.shape = (1 * num)
    CH2 = np.square(CH)
    rate = np.zeros((len(CH), 2))
    choice = random_N(r_prob, c_prob, len(CH))
    for i in range(len(CH)):
        time = i * 0.1
        r_power = power * CH2[i]
        rate[i][0] = np.round(time, 2)
        rate[i][1] = np.round(choice[i] * fc * np.log(1 + np.divide(r_power, choice[i] * fc * noise)) / np.log(2.0) / 1000000, 6)   # 换底公式,顺手将bps转换为MBPS

    return rate


def random_N(r_prob, c_prob, num):
    temp = []
    r = tf.random.categorical(tf.math.log(r_prob), np.ceil(num / (r_change * 10)))[0]
    # Tensor转数字
    c = tf.random.categorical(tf.math.log(c_prob), np.ceil(num / (c_change * 10)))[0]
    r = np.array(r, dtype=int)
    c = np.array(c, dtype=int)
    index_c = -1
    index_r = -1
    for i in range(num):
        if i % 10 == 0:
            index_c = index_c + 1
        if i % 100 == 0:
            index_r = index_r + 1
        temp.append(N_choice[r[index_r]][c[index_c]])
    # print(temp)
    return temp


def generate_trace(num_file, time_length, filepath):
    if os.path.exists(filepath):
        os.system("rm -rf " + filepath)
    os.system("mkdir " + filepath)
    for i in range(num_file):
        data = generate_CSI(time_length)
        temp_r = i % len(r_pro)
        temp_c = i % len(c_pro)
        # a = [r_pro[temp_r]]
        rate = calc_rate(data, tf.constant([r_pro[temp_r]]), tf.constant([c_pro[temp_c]]))
        # os.system("rm -rf " + filepath + str(i))
        # data_write([time_length], "./dateset/" + str(i))
        data_write(rate, filepath + '/trace_' + str(i))


def generate_trace_special(time_length):
    data = generate_CSI(time_length)
    # a = [r_pro[temp_r]]
    rate = calc_rate(data, tf.constant([[0.25, 0.25, 0.25, 0.25]]), tf.constant([[0.1, 0.8, 0.1]]))
    os.system("rm -rf " + "./test_dateset/" + "special_trace")
    # data_write([time_length], "./dateset/" + str(i))
    data_write(rate, "./test_dateset/" + "special_trace")
def main():
    print("generate_dateset is done.")


if __name__ == '__main__':
    # generate_trace_special(60)
    # index = [i for i in range(10) if i % 2 == 0]
    generate_trace(9, 60, 'test_dateset/trace')
    generate_trace(100, 2000, 'dateset/')
    # main()
    # # print(np.log(2))
    # result = []
    # data = generate_CSI(2000)
    # rate = calc_rate(data, tf.constant([[0., 0., 0., 1.]]), tf.constant([[0.1, 0.8, 0.1]]))
    # x = rate[:, 0]
    # y = rate[:, 1]
    # y = np.reshape(y, (-1, 100))
    # mean = np.mean(y, axis=1)
    # index = [i for i in range(len(mean))]
    # plt.plot(index, mean)
    # plt.show()
