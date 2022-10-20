import os

from matplotlib import pyplot as plt

from data_load_fix import data_write, data_load
import numpy as np
import itertools
import load_trace


VIDEO_BIT_RATE = [1000, 2000, 3000, 4000, 4750]  # Kbps
A_DIM = 5
MPC_FUTURE_CHUNK_COUNT = 5
RANDOM_SEED = 69
CHUNK_COMBO_OPTIONS = []
TRAIN_TRACES = './dateset/'
all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)


def MPC_test(trace_path):
    import env as env
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
    #  test_chunk中指定了视频
    env.test_chunk(cooked_time, cooked_bw)

    result = []
    bandwidth = []
    bitrate_choice = []
    buffer_status = []
    rebuf_status = []
    reward_record = []

    # np.random.seed(RANDOM_SEED)
    # MPC算法所需要的两个数组记录
    past_bandwidth_ests = []
    past_errors = []
    for combo in itertools.product(range(A_DIM), repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)
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
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / 10  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / 1000.0  # kilo byte / ms
        # 转换成10s一格
        state[3, -1] = float(rebuf) / 10.0  # 10 sec
        state[4, :5] = np.array(next_video_chunk_sizes) / 1000.0 / 1000.0  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, env.TOTAL_VIDEO_CHUNCK) / float(env.TOTAL_VIDEO_CHUNCK)

        bandwidth.append(state[2, -1] * 8 * 1000)
        bitrate_choice.append(VIDEO_BIT_RATE[bit_rate])
        buffer_status.append(state[1, -1] * 10)
        rebuf_status.append(rebuf)

        # MPC算法开始
        if len(past_bandwidth_ests) > 0:
            curr_error = abs(past_bandwidth_ests[-1] - state[2, -1]) / float(state[2, -1])
        else:
            curr_error = 0
        past_errors.append(curr_error)

        # 去除过去5个时间点的带宽
        past_bandwidth = state[2, -5:]
        # 只保存不是零的数值
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        # 开始计算调和平均数（harmonic mean），就是倒数的平均数的倒数
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth))
        error_pos = -5
        if len(past_errors) < 5:
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        # 第二个时间点的带宽估算肯定是和第一个时间点的带宽一样
        future_bandwidth = harmonic_bandwidth / (1 + max_error)
        past_bandwidth_ests.append(harmonic_bandwidth) # 带宽的估算已经完毕


        last_index = int(env.TOTAL_VIDEO_CHUNCK - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if video_chunk_remain < MPC_FUTURE_CHUNK_COUNT:
            # print(future_chunk_length)
            future_chunk_length = video_chunk_remain

        max_reward = -1000000
        best_combo = ()
        start_buffer = buffer_size
        # CHUNK_COMBO_POTIONS的大小就是5的7次方，5的MPC_FUTURE_CHUNK_COUNT方
        # 按照当前预估的带宽当作未来MPC_FUTURE_CHUNK_COUNT个时间点的带宽去计算总的reward
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            # 头一次就是记录了上次的bit_rate，bitrate变量一直是没有动的，用了新的变量来做记录
            last_quality = int(bit_rate)
            # combo就是full_combo，full_combo遍历每一种可能选择
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position
                download_time = (env.get_video_chunk_size(chunk_quality, index)) / (future_bandwidth * 1000000.0)
                if curr_buffer < download_time:
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                last_quality = chunk_quality
            # 看未来MPC_FUTURE_CHUNK_COUNT步长的奖励值的总和
            reward = (bitrate_sum / 1000.) - (10 * curr_rebuffer_time) - (smoothness_diffs / 1000.)

            if reward > max_reward:
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                send_data = 0
                if best_combo != ():
                    send_data = best_combo[0]

        bit_rate = send_data
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            env.get_video_chunk(bit_rate)
        done = end_of_video
        reward = VIDEO_BIT_RATE[bit_rate] / 1000.0 \
                 - 10 * rebuf \
                 - 0.1 * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                VIDEO_BIT_RATE[last_bit_rate]) / 1000.0
        epoch_reward += reward
        reward_record.append(reward)
        if done:
            result.append(epoch_reward)
            break
        # print(env.video_chunk_counter)


    print('done')
    print(np.sum(result))
    index = [i for i in range(len(bandwidth))]

    plt.plot(index, bandwidth, color='red')

    plt.plot(index, bitrate_choice)
    plt.ylabel(trace_path)
    plt.show()
    plt.plot(index, buffer_status)
    plt.show()
    plt.plot(index, rebuf_status)
    plt.show()
    print(result)
    return np.sum(result)


if __name__ == '__main__':
    result = []
    test_dateset_path = 'test_dateset/trace/'
    cooked_files = os.listdir(test_dateset_path)
    # train_test()
    for cooked_file in cooked_files:
        file_path = test_dateset_path + cooked_file
        result.append(MPC_test(file_path))
    print(f'test中的reward平均10个总和为：{np.sum(result) / len(result)}')
    data_write(np.reshape(np.sum(result) / len(result), (1, -1)), './MPCtest_log')
