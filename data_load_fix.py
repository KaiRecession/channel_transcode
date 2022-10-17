# -*- coding: UTF-8 -*-
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt


def data_load2(file_path):
    data = np.loadtxt(file_path, dtype=np.str, delimiter=" ")
    return data


def data_load(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    data_array = np.array(data)
    data_array = data_array.squeeze(1)
    return data_array


def data_merge(file1, file2, outfile):
    # pd.merge(file1, file2)
    data1 = pd.read_csv(file1, delimiter=' ', header=None)
    data2 = pd.read_csv(file2, delimiter=' ', header=None)
    data_write(data1, outfile)
    data_write(data2, outfile)


def data_write(data, file_path):
    data = np.array(data)
    # data = np.reshape(data, (-1, 6))
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(data)

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(file_path, index=False, sep=' ', header=None, mode='a')


if __name__ == '__main__':
    data = data_load('real_record/record1.csv')
    data = np.reshape(data, (-1, 10))
    a = []
    b = []
    length = len(data)
    for i in range(len(data)):
        if data[i] > 0:
            a.append(i)
            b.append(data[i])
    plt.figure(figsize=(16, 9))  # 分辨率为：1600 * 900
    plt.plot(a, b, label='2-5 Mbps')
    # plt.plot(a, filesize(b, d), marker='x', color='blue', label='3Mbps')
    # plt.plot(a, filesize(b, e), marker='x', color='green', label='2Mbps')
    # plt.plot(a, filesize(b, f), marker='x', color='purple', label='1Mbps')
    plt.legend()
    plt.ylabel('reward')
    plt.xlabel('epoch')
    # plt.bar(a, b, width=0.05, color='red')
    # plt.bar(a, b, width=1)
    # plt.scatter(a, b, s=10, c='red')
    plt.show()
    # print(np.max(data))
    # print(np.mean(data))
    # data_write([[1, 2], [2, 2]], 'test1.csv')
    # data_write([[1], [2]], 'test2.csv')
    # data_merge('test1.csv', 'test2.csv', 'test3.csv')
    # print('woc')
