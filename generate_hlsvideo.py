import os

import numpy as np
import pandas as pd

random_seed = 10
num_video = 100
max_num_bitrate = 5
min_num_bitrate = 0
max_num_chunk = 15 * 60
min_num_chunk = 15 * 5
length_video = 4
VIDEO_BIT_RATE = [1000, 2000, 3000, 4000, 4750]
size_noise = 0.12

np.random.seed(random_seed)


def data_load(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    data_array = np.array(data)
    return data_array


def data_write(data, file_path):
    data = np.array(data)
    dataframe = pd.DataFrame(data)
    # dataframe.to_csv(file_path, index=False, sep=' ', header=None, mode='a')
    dataframe.to_csv(file_path, index=False, sep=' ', header=None)


def main():
    for i in range(num_video):
        num_chunks = np.random.randint(min_num_chunk, max_num_chunk)
        video_sub = []
        for item in VIDEO_BIT_RATE:
            size = []
            for _ in range(num_chunks):
                noise = np.random.normal(1, size_noise)
                size.append(round((item / 8.0 * length_video * noise)))
            video_sub.append(size)
        # print("end")
        data_write(video_sub, './videos/hls_' + i.__str__())
        # test = data_load('./videos/hls_' + i.__str__())
        # print(test == np.array(video_sub))


if __name__ == '__main__':
    os.system("rm -rf " + "./videos")
    os.system("mkdir " + "./videos")
    # data_write([time_length], "./dateset/" + str(i))
    main()

