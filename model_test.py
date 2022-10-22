import os

from matplotlib import pyplot as plt

import load_trace
import tensorflow as tf
import numpy as np

from A3C import ActorCritic
from data_load_fix import data_write, data_load
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(1231)
np.random.seed(1231)
TRAIN_TRACES = './dateset/'
all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
VIDEO_BIT_RATE = [1000, 2000, 3000, 4000, 4750]
model_weight = 'model/v2.ckpt'
from tensorflow.keras import layers, optimizers, Model
server = ActorCritic(6, 5)
server.load_weights(model_weight)

# 随机挑选一个轨迹和视频
def train_test():
        import env
        # server = tf.saved_model.load('./model/0')
        env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=1)
        result = []
        for epi_counter in range(2):
            # print(f'epi_counter: {epi_counter}')
            # 初始化的时候设置值
            last_bit_rate = 1
            bit_rate = 1
            state = np.zeros((6, 8))
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                env.get_video_chunk(bit_rate)
            epoch_reward = 0.
            epoch_steps = 0
            done = False
            while not done:
                # dequeue history record
                # 记录回滚历史状态state
                state = np.roll(state, -1, axis=1)
                # this should be S_INFO number of terms
                # 除了剩余块的那一行，剩下的都是记录了过去的K个信息的，好家伙
                state[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / 10  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / 1000.0  # kilo byte / ms
                # 转换成10s一格
                state[3, -1] = float(rebuf) / 10.0  # 10 sec
                state[4, :5] = np.array(next_video_chunk_sizes) / 1000.0 / 1000.0  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, env.TOTAL_VIDEO_CHUNCK) / float(
                    env.TOTAL_VIDEO_CHUNCK)
                current_state = np.reshape(state, (1, 6, 8))

                logits, _ = server(tf.constant(current_state, dtype=tf.float32))
                # print(logits)
                probs = tf.nn.softmax(logits)
                # print(probs)
                # 按照概率选择action, np.random.choice中的5代表从0-5筛选
                bit_rate = np.random.choice(5, p=probs.numpy()[0])
                # print(action)
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)
                reward = VIDEO_BIT_RATE[bit_rate] / 1000.0 \
                         - 4.3 * rebuf \
                         - 1 * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[last_bit_rate]) / 1000.0
                done = end_of_video
                # 相当于word里面的一次轨迹的reward总和，就是为了方便展示信息
                epoch_reward += reward
                # 这个step是本次轨迹走过的步数
                epoch_steps += 1
                last_bit_rate = bit_rate

                new_state = current_state
                if done:
                    result.append(epoch_reward)
                    break
        print(f'test中的reward平均10个总和为：{np.sum(result)}, video_id:{env.video_idx}, trace_id:{env.trace_idx}')
        results = []
        results.append(np.sum(result))
        results.append(env.video_idx)
        results.append(env.trace_idx)
        data_write(np.reshape(results, (1, -1)), './test_log')


# 挑选指定的视频
def train_test_plot(trace_path):
    import env
    # server = tf.saved_model.load('./model/0')
    env = env.Environment(all_cooked_time=all_cooked_time,
                          all_cooked_bw=all_cooked_bw,
                          random_seed=1)

    cooked_time = []
    cooked_bw = []
    # print file_path
    with open(trace_path, 'rb') as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1]))
    #  test_chunk中不仅指定了轨迹，同时指定了视频
    env.test_chunk(cooked_time, cooked_bw)


    result = []
    bandwidth = []
    bitrate_choice = []
    buffer_status = []
    rebuf_status = []
    reward_record = []
    for epi_counter in range(1):
        # print(f'epi_counter: {epi_counter}')
        # 初始化的时候设置值
        last_bit_rate = 1
        bit_rate = 1
        state = np.zeros((6, 8))
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            env.get_video_chunk(bit_rate)
        epoch_reward = 0.
        epoch_steps = 0
        done = False
        while not done:
            # dequeue history record
            # 记录回滚历史状态state
            state = np.roll(state, -1, axis=1)
            # this should be S_INFO number of terms
            # 除了剩余块的那一行，剩下的都是记录了过去的K个信息的，好家伙
            state[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / 10  # 10 sec
            # 过去时间点的带宽
            state[2, -1] = float(video_chunk_size) / float(delay) / 1000.0  # kilo byte / ms
            # 转换成10s一格
            state[3, -1] = float(rebuf) / 10.0  # 10 sec
            state[4, :5] = np.array(next_video_chunk_sizes) / 1000.0 / 1000.0  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, env.TOTAL_VIDEO_CHUNCK) / float(env.TOTAL_VIDEO_CHUNCK)
            current_state = np.reshape(state, (1, 6, 8))
            bandwidth.append(state[2, -1] * 8 * 1000)
            bitrate_choice.append(VIDEO_BIT_RATE[bit_rate])
            buffer_status.append(state[1, -1] * 10)
            rebuf_status.append(rebuf)

            logits, _ = server(tf.constant(current_state, dtype=tf.float32))
            # print(logits)
            probs = tf.nn.softmax(logits)
            # print(probs)
            # 按照概率选择action, np.random.choice中的5代表从0-5筛选
            bit_rate = np.random.choice(5, p=probs.numpy()[0])
            # print(action)
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)
            reward = VIDEO_BIT_RATE[bit_rate] / 1000.0 \
                     - 10 * rebuf \
                     - 0.1 * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                  VIDEO_BIT_RATE[last_bit_rate]) / 1000.0
            done = end_of_video
            # 相当于word里面的一次轨迹的reward总和，就是为了方便展示信息
            epoch_reward += reward
            reward_record.append(reward)
            # 这个step是本次轨迹走过的步数
            epoch_steps += 1
            last_bit_rate = bit_rate
            new_state = current_state
            if done:
                result.append(epoch_reward)
                break
    # print(f'test中的reward平均10个总和为：{np.sum(result)}, video_id:{env.video_idx}, trace_id:{env.trace_idx}')


    results = []
    results.append(np.sum(result))
    results.append(env.video_idx)
    results.append(env.trace_idx)
    # data_write(np.reshape(results, (1, -1)), './test_log')
    # # 画图部分，测试时注释掉
    # index = [i for i in range(len(bandwidth))]
    #
    # plt.plot(index, bandwidth, color='red')
    #
    # plt.plot(index, bitrate_choice)
    # plt.ylabel(trace_path)
    # plt.show()
    # plt.plot(index, buffer_status)
    # plt.show()
    # plt.plot(index, rebuf_status)
    # plt.show()
    # print(result)
    return np.sum(result)


if __name__ == '__main__':
    result = []
    test_dateset_path = 'test_dateset/trace/'
    cooked_files = os.listdir(test_dateset_path)
    # train_test()
    for cooked_file in cooked_files:
        file_path = test_dateset_path + cooked_file
        result.append(train_test_plot(file_path))
    mean = np.sum(result) / len(result)
    print(f'test中的reward平均10个总和为：{mean}')
    # print(result)
    data_write(np.reshape(mean, (1, -1)), './test_log')
    best = data_load('temp_best_result')[0]
    if mean > best:
        os.system("rm -rf " + "temp_best_result")
        data_write([mean], 'temp_best_result')
        server.save_weights('best_model/v2.ckpt')

