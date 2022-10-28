num_core = 8
# ms
max_delay_tc = 1315
# Mbps
Wom = 300 * pow(10, 6) / 8
# Watt/bit
eom = 8 * pow(10, -6)
wcm = 8 * pow(10, -8)
p0 = 1 * pow(10, -9)
ctm = 300


class Energy:
    def __init__(self):
        self.buffer_transcode = {}

    # 设置转码的buffer的初始值
    def set_buffer_transcode(self, env, index_length, bitrate):
        for i in range(index_length):
            index = env.video_chunk_counter + i
            if index >= env.TOTAL_VIDEO_CHUNCK:
                break
            cache_bitrate = env.cache_status[index]
            if cache_bitrate != -1:
                chunk_size = env.get_video_chunk_size(cache_bitrate, index)
                # value放时间值更好
                # TODO 在buffer里面放转码的(延时，能耗)，key设置成视频块的索引，cachebitrate不存在就设置成-1
                self.buffer_transcode[index] = (self.cacal_energy_transcode(chunk_size, bitrate, cache_bitrate))
            else:
                self.buffer_transcode[index] = (0.0, 0.0)

    def modify_buffer_transcode(self, counter, delay):
        # print(f'counter{counter}')
        temp = delay
        index = counter
        # print(111)

        while temp >= 0 and index in self.buffer_transcode:
            if temp > self.buffer_transcode[index][1]:
                temp -= self.buffer_transcode[index][1]
                self.buffer_transcode[index] = (self.buffer_transcode[index][0], 0)
                index += 1
            else:
                temp = -1
                self.buffer_transcode[index] = (self.buffer_transcode[index][0], self.buffer_transcode[index][1] - temp)

    def cacal_energy(self, bitrate, env, cache_bitrate):
        energy = 0
        Eom = 0
        Tom = 0
        Ec = 0
        Etc = 0
        delay_tc = 0
        if cache_bitrate == -1:
            Eom, Tom = self.cacal_energy_origin(env.get_video_chunk_size(bitrate, env.video_chunk_counter))
            Ec = self.cacal_energy_cache(env.get_video_chunk_size(bitrate, env.video_chunk_counter))
        else:
            Etc, delay_tc = self.buffer_transcode[env.video_chunk_counter]
        T = Tom + delay_tc
        energy = Eom + Ec + Etc
        return energy, T

    def cacal_energy_origin(self, chunk_size):
        Tom = chunk_size / Wom
        Eom = eom * chunk_size * Tom
        return Eom, Tom

    def cacal_energy_cache(self, chunk_size):
        Ec = wcm * chunk_size
        return Ec

    def cacal_energy_transcode(self, chunk_size, bitrate, cache_bitrate):
        if bitrate >= cache_bitrate:
            delay_tc = 0
        else:
            delay_tc = (max_delay_tc - (4 - cache_bitrate) * 200) * (1 - (cache_bitrate - bitrate - 1) * 0.25)
        Etc = p0 * ctm * chunk_size * delay_tc * pow(10, -3)
        return Etc, delay_tc


def main():

    test = Energy()
    # energy, T = test.cacal_energy(2380108, 0, 3, True)
    print('done')


if __name__ == '__main__':
    main()
