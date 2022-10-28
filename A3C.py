import multiprocessing
import os
import threading
import env as env
import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from queue import Queue
from tensorflow.keras import layers, optimizers
from tensorflow.keras import Model
from sklearn import model_selection

import load_trace

plt.rcParams['font.size'] = 18
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = [9, 7]
plt.rcParams['font.family'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(1231)
np.random.seed(1231)
TRAIN_TRACES = './dateset/'
all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
VIDEO_BIT_RATE = [1000, 2000, 3000, 4000, 4750]  # Kbps
model_weight = None


class ActorCritic(Model):
    def __init__(self, state_size, action_size, action2_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = layers.Dense(128, activation='relu')
        self.conv1 = layers.Conv1D(filters=128, kernel_size=4, activation='relu')
        self.conv2 = layers.Conv1D(128, 4, activation='relu')
        self.conv3 = layers.Conv1D(128, 4, activation='relu')
        self.conv4 = layers.Conv1D(128, 4, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(32, activation='relu')
        self.fc4 = layers.Dense(action_size)
        self.fc5 = layers.Dense(action2_size)

        self.fc21 = layers.Dense(128, activation='relu')
        self.conv21 = layers.Conv1D(128, 4, activation='relu')
        self.conv22 = layers.Conv1D(128, 4, activation='relu')
        self.conv23 = layers.Conv1D(128, 4, activation='relu')
        self.conv24 = layers.Conv1D(128, 4, activation='relu')
        self.fc22 = layers.Dense(128, activation='relu')
        self.fc23 = layers.Dense(32, activation='relu')
        self.fc24 = layers.Dense(1)

    def call(self, inputs):
        # print(inputs)
        # 还是定义了两个小网络，上面是actor，下面是critic
        se = tf.shape(inputs)[0]
        x1 = self.fc1(inputs[:, 0:1, -1])
        x1 = tf.reshape(x1, (se, -1))
        # print(inputs[:, 1:2, :])
        x2 = self.conv1(tf.reshape(inputs[:, 1:2, :], (se, -1, 1)))
        x2 = tf.reshape(x2, (se, -1))
        x3 = self.conv2(tf.reshape(inputs[:, 2:3, :], (se, -1, 1)))
        x3 = tf.reshape(x3, (se, -1))
        x4 = self.conv3(tf.reshape(inputs[:, 3:4, :], (se, -1, 1)))
        x4 = tf.reshape(x4, (se, -1))
        x5 = self.conv4(tf.reshape(inputs[:, 4:5, :5], (se, -1, 1)))
        x5 = tf.reshape(x5, (se, -1))
        x6 = self.fc2(inputs[:, 4:5, -1])
        x6 = tf.reshape(x6, (se, -1))
        x7 = tf.concat([x1, x2, x3, x4, x5, x6], axis=1)
        # x8 = self.fc3(x7)
        logits = self.fc4(x7)
        logits_length = self.fc5(x7)
        # print(logits)

        x21 = self.fc21(inputs[:, 0:1, -1])
        x21 = tf.reshape(x21, (se, -1))
        x22 = self.conv21(tf.reshape(inputs[:, 1:2, :], (se, -1, 1)))
        x22 = tf.reshape(x22, (se, -1))
        x23 = self.conv22(tf.reshape(inputs[:, 2:3, :], (se, -1, 1)))
        x23 = tf.reshape(x23, (se, -1))
        x24 = self.conv23(tf.reshape(inputs[:, 3:4, :], (se, -1, 1)))
        x24 = tf.reshape(x24, (se, -1))
        x25 = self.conv24(tf.reshape(inputs[:, 4:5, :5], (se, -1, 1)))
        x25 = tf.reshape(x25, (se, -1))
        x26 = self.fc22(inputs[:, 4:5, -1])
        x26 = tf.reshape(x26, (se, -1))
        x27 = tf.concat([x21, x22, x23, x24, x25, x26], axis=1)
        # x28 = self.fc23(x27)
        values = self.fc24(x27)
        # print(values)

        return logits, logits_length, values


# def record(epoch, epoch_reward, worker_id, global_epoch_reward, result_queue, total_loss, num_steps):
#     if global_epoch_reward == 0:
#         global_epoch_reward = epoch_reward
#     else:
#         global_epoch_reward = global_epoch_reward * 0.99 + epoch_reward * 0.01
#     print(
#         f"{epoch} | "
#         f"Average Reward: {int(global_epoch_reward)} | "
#         f"Episode Reward: {int(epoch_reward)} | "
#         f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
#         f"Steps: {num_steps} | "
#         f"Worker: {worker_id}"
#     )
#     result_queue.put(global_epoch_reward)  # 保存回报，传给主线程
#     return global_epoch_reward


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.lengths = []
        self.rewards = []

    def store(self, state, action, length, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.lengths.append(length)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.lengths = []


class Agent:
    def __init__(self):
        self.opt = optimizers.Adam(1e-3)
        self.server = ActorCritic(6, 5, 5)
        self.server(tf.random.normal((1, 6, 8)))
        if model_weight != None:
            self.server.load_weights(model_weight)

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.server, self.opt, res_queue, i)
                   for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print("starting worker{}".format(i))
            worker.start()
        returns = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                returns.append(reward)
            else:
                break
        [w.join() for w in workers]

        # 等待所有work线程结束，每个word线程的训练次数有限制的
        print(returns)

        plt.figure()
        plt.plot(np.arange(len(returns)), returns)
        plt.xlabel('epochs')
        plt.ylabel('scores')
        plt.savefig('demo08.svg')


class Worker(threading.Thread):
    def __init__(self, server, opt, result_queue, id):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.server = server
        self.opt = opt
        self.client = ActorCritic(6, 5, 5)
        self.worker_id = id
        # self.env = gym.make('CartPole-v1').unwrapped
        self.env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=1231)
        self.ep_loss = 0.0

    def run(self):
        # 相当于存储一整条轨迹的类
        mem = Memory()
        # 每个worker线程的epoch次数
        for epi_counter in range(50000):
            # print(f'epi_counter: {epi_counter}')
            # 初始化的时候设置值
            last_bit_rate = 1
            bit_rate = 1
            state = np.zeros((6, 8))
            # 1、设置初始化的length长度
            self.env.energy.set_buffer_transcode(self.env, 1, bit_rate)
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, energy = \
                self.env.get_video_chunk(bit_rate)
            # 忽略第一次的延时
            # print(42432143)
            rebuf = 0.0
            mem.clear()
            epoch_reward = 0.
            epoch_steps = 0
            done = False
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
            state[5, -1] = np.minimum(video_chunk_remain, self.env.TOTAL_VIDEO_CHUNCK) / float(
                self.env.TOTAL_VIDEO_CHUNCK)
            current_state = np.reshape(state, (1, 6, 8))
            while not done:


                logits, logits_length, _ = self.client(tf.constant(current_state, dtype=tf.float32))
                # print(logits)
                probs = tf.nn.softmax(logits)
                # print(probs)
                # 按照概率选择action, np.random.choice中的5代表从0-5筛选
                bit_rate = np.random.choice(5, p=probs.numpy()[0])
                # print(action)

                probs = tf.nn.softmax(logits_length)
                # print(probs)
                # 按照概率选择action, np.random.choice中的5代表从0-5筛选
                length = np.random.choice(5, p=probs.numpy()[0]) + 1
                # 2、经过网络选择后设置未来的长度
                self.env.energy.set_buffer_transcode(self.env, length + 1, bit_rate)
                # print(f'self.env.energy.buffer_transcode{self.env.energy.buffer_transcode}, length{length}')
                # print(length)

                for _ in range(length):
                    # print(f'work_id{self.worker_id}, done {done}')
                    delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, energy = self.env.get_video_chunk(bit_rate)
                    reward = VIDEO_BIT_RATE[bit_rate] / 1000.0 \
                             - 4.75 * rebuf \
                             - 0.1 * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / 1000.0
                    done = end_of_video
                    # 相当于word里面的一次轨迹的reward总和，就是为了方便展示信息
                    epoch_reward += reward
                    # 存储的state和action必须是同一对输入输出，reward也是这次action的reward
                    mem.store(current_state, bit_rate, length, reward)
                    # 这个step是本次轨迹走过的步数
                    epoch_steps += 1
                    last_bit_rate = bit_rate


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
                    state[5, -1] = np.minimum(video_chunk_remain, self.env.TOTAL_VIDEO_CHUNCK) / float(
                        self.env.TOTAL_VIDEO_CHUNCK)
                    current_state = np.reshape(state, (1, 6, 8))
                    new_state = current_state

                    if done:
                        break

                if done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem)
                    # worker线程的梯度拿出来
                    grads = tape.gradient(total_loss, self.client.trainable_variables)
                    # 把worker线程的梯度给server更新参数
                    self.opt.apply_gradients(zip(grads, self.server.trainable_variables))
                    # 把server参数更新到client
                    self.client.set_weights(self.server.get_weights())
                    mem.clear()
                    self.result_queue.put(epoch_reward)
                    print("epoch=%s," % epi_counter, "worker=%s," % self.worker_id, "reward=%s" % epoch_reward)
                    if self.worker_id == 0:
                        # tf.saved_model.save(self.server, 'model/0')
                        self.server.save_weights('model/v2.ckpt')
                        os.system('python model_test.py')
                    break
        # 放入None表示结束
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        # 如果进来的state不是因为done，那么最后一步的之后的期望reward就用网络计算
        if done:
            reward_sum = 0.
        else:
            # 把网络输出的value变成一个普通常量
            reward_sum = self.client(tf.constant(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum / self.env.TOTAL_VIDEO_CHUNCK
            discounted_rewards.append(reward_sum)
        # 倒着计算reward后，把列表反转，回到正常的顺序
        discounted_rewards.reverse()
        # print(tf.constant(np.vstack(memory.states)))
        # 此时得到的概率，值都是批次所得
        logits, logits_length, values = self.client(tf.constant(np.vstack(memory.states), dtype=tf.float32))
        # 计算advantage = R() - v(s)
        advantage = tf.constant(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        value_loss = advantage ** 2
        policy = tf.nn.softmax(logits)
        # 计算多分类的交叉熵
        # 就是一步计算了-log对应的概率，直接让loss函数连负号都不用加
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss = policy_loss * tf.stop_gradient(advantage)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        # 保留一些试探性的选择，输出选项的各个概率值之间差距越大，entropy就越小，适当增大entropy，保留一些试探性的选择
        policy_loss = policy_loss - 0.01 * entropy

        policy_length = tf.nn.softmax(logits_length)
        policy_loss_length = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.lengths, logits=logits_length)
        policy_loss_length = policy_loss_length * tf.stop_gradient(advantage)
        entropy_length = tf.nn.softmax_cross_entropy_with_logits(labels=policy_length, logits=logits_length)
        policy_loss_length = policy_loss_length - 0.01 * entropy_length

        total_loss = tf.reduce_mean((0.5 * value_loss + (policy_loss + policy_loss_length) / 2))
        return total_loss


if __name__ == '__main__':
    master = Agent()
    master.train()
