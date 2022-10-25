import numpy as np

from data_load_fix import data_load
from matplotlib import pyplot as plt


def main():
    best = data_load('test_log')
    print(len(best))
    index = [i for i in range(len(best))]
    MPC = [254, 114, 308, 252, 331, 300, 309, 331, 31]
    mean = np.mean(MPC)
    MPC_base = [mean for i in range(len(best))]
    plt.plot(index, best)
    plt.plot(index, MPC_base, color='red')
    plt.show()
    print(best)


if __name__ == '__main__':
    main()
