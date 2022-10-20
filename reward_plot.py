from data_load_fix import data_load
from matplotlib import pyplot as plt


def main():
    best = data_load('test_log')
    print(len(best))
    index = [i for i in range(len(best))]
    plt.plot(index, best)
    plt.show()
    print(best)

if __name__ == '__main__':
    main()
