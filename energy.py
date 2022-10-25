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


def cacal_energy(chunk_size, bitrate, cache_bitrate, flag_origin):
    energy = 0
    Eom = 0
    Tom = 0
    Ec = 0
    Etc = 0
    delay_tc = 0
    if flag_origin is True:
        Eom, Tom = cacal_energy_origin(chunk_size)
        Ec = cacal_energy_cache(chunk_size)
    else:
        Etc, delay_tc = cacal_energy_transcode(chunk_size, bitrate, cache_bitrate)
    T = Tom + delay_tc
    energy = Eom + Ec + Etc
    return energy, T


def cacal_energy_origin(chunk_size):
    Tom = chunk_size / Wom
    Eom = eom * chunk_size * Tom
    return Eom, Tom


def cacal_energy_cache(chunk_size):
    Ec = wcm * chunk_size
    return Ec


def cacal_energy_transcode(chunk_size, bitrate, cache_bitrate):
    if bitrate >= cache_bitrate:
        delay_tc = 0
    else:
        delay_tc = (max_delay_tc - (4 - cache_bitrate) * 200) * (1 - (cache_bitrate - bitrate - 1) * 0.25)
    Etc = p0 * ctm * chunk_size * delay_tc * pow(10, -3)
    return Etc, delay_tc


def main():
    energy, T = cacal_energy(2380108, 0, 3, True)
    print('done')


if __name__ == '__main__':
    main()
