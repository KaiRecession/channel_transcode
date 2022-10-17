# -*- coding: UTF-8 -*-
import os
import matplotlib.pyplot as plt

# 拿到每一个轨迹的时间点和对应的带宽bw，顺带返回了一个文件名字的列表，总共三个参数
import numpy as np

dataset = './dateset/'


def load_trace(cooked_trace_folder=dataset):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


if __name__ == '__main__':
    all_cooked_time, all_cooked_bw, all_file_names = load_trace()
    a = all_cooked_time[0]
    b = all_cooked_bw[0]
    plt.plot(a, b)
    plt.ylabel('Bit rate (Mbps)')
    plt.xlabel('time (s)')
    # plt.bar(a, b, width=0.05, color='red')
    # plt.bar(a, b, width=1)
    # plt.scatter(a, b, s=10, c='red')
    plt.show()
    print('结束')