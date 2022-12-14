# -*- coding: UTF-8 -*-
import os
import load_trace
import numpy as np

from energy import Energy

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 5
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './videos/'
TRAIN_TRACES = './dateset/'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.energy = Energy()

        # pick a random trace file
        # 挑选一个随机的网络track
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        # 根据文件加载里面的track信息
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        # 从这一条track中随机挑选一个时间点开始，当到末尾了从0开始
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_files = os.listdir(VIDEO_SIZE_FILE)
        self.video_idx = 0
        self.video_size = {}  # in bytes
        # 加入cache状态
        self.cache_status = []
        self.TOTAL_VIDEO_CHUNCK = 0
        # 加载不同视频文件的大小
        self.set_video_size()

    def set_video_size(self):
        self.video_idx = np.random.randint(0, len(self.video_files))
        file_path = VIDEO_SIZE_FILE + self.video_files[self.video_idx]
        bitrate = 0
        with open(file_path, 'rb') as f:
            for line in f:
                self.video_size[bitrate] = np.array(line.split(), dtype=int).tolist()
                bitrate += 1
        self.TOTAL_VIDEO_CHUNCK = len(self.video_size[0])
        for i in range(self.TOTAL_VIDEO_CHUNCK):
            self.cache_status.append(np.random.randint(-1, 5))

    # MPC专用函数
    def get_video_chunk_size(self, quality, index):
        return self.video_size[quality][index]

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        
        while True:  # download video chunk over mahimahi
            # 根据随机开始的时间点，将带宽转化
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            # 记录时间差
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
            # 计算下载了多少，随后是一个参数，不用在意
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                # 视频块的下载不会正好的用完每一段时间，最后一段不是整数，计算碎片时间
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                # 视频块下载完毕，跳出
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1
            # 到轨迹末尾的时候，归零
            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
        # 循环结束，记录下载视频块的时间
        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
        # 修改转码buffer
        self.energy.modify_buffer_transcode(self.video_chunk_counter + 1, delay)
        # rebuffer time
        # 计算延迟时间，buffer是已经下载的视频时长
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        # buffer在下载视频的时候播放了多长的视频，就减少了多长时间的buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        # 下载来的视频又增大了buffer
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        # 大于buffer值就停止下载
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            # 计算大于等于改值的最小整数
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            # 等待睡眠时间，把buffer消耗一点
            self.buffer_size -= sleep_time
            self.energy.modify_buffer_transcode(self.video_chunk_counter + 1, sleep_time)
            # 把带宽轨迹的时间点跳过睡眠的时间点
            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size
        # 计算能耗和时延
        temp_energy, temp_delay = self.energy.cacal_energy(quality, self, self.cache_status[self.video_chunk_counter])
        delay += temp_delay

        self.video_chunk_counter += 1
        video_chunk_remain = self.TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
            self.set_video_size()

        next_video_chunk_sizes = []
        next_video_chunk_buffer_status = []
        for i in range(BITRATE_LEVELS):
            if (self.video_chunk_counter + i < self.TOTAL_VIDEO_CHUNCK):
                next_video_chunk_sizes.append(self.video_size[4][self.video_chunk_counter + i])
                next_video_chunk_buffer_status.append(self.cache_status[self.video_chunk_counter + i])
            else:
                next_video_chunk_sizes.append(0.0)
                next_video_chunk_buffer_status.append(0)

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain, \
            temp_energy, \
            next_video_chunk_buffer_status

    # 设置了指定的网络轨迹之后同时默认指定了相应的指定测试视频块
    def test_chunk(self, cooked_time, cooked_bw):
        self.cooked_time = cooked_time
        self.cooked_bw = cooked_bw
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = 0

        file_path = 'test_dateset/special_videos'
        bitrate = 0
        with open(file_path, 'rb') as f:
            for line in f:
                self.video_size[bitrate] = np.array(line.split(), dtype=int).tolist()
                bitrate += 1
        self.TOTAL_VIDEO_CHUNCK = len(self.video_size[0])

def ftest():
    video_files = os.listdir(VIDEO_SIZE_FILE)
    for video_file in video_files:
        file_path = VIDEO_SIZE_FILE + video_file
        video_size = {}
        bitrate = 0
        with open(file_path, 'rb') as f:
            for line in f:
                video_size[bitrate] = np.array(line.split(), dtype=int).tolist()
                bitrate += 1
    print("end")


if __name__ == '__main__':
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    net_env = Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=1)
    net_env.set_buffer_transcode(5, 2)
    while True:
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(0)
        net_env.set_buffer_transcode(3, 2)
        print(134)

    print(net_env)
