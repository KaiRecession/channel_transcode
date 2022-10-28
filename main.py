import numpy as np
import tensorflow


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    s_batch = [np.zeros((6, 8))]
    hashtest = {}
    hashtest[1] = 1
    hashtest[3] = 2
    print(hashtest.get(1))

    a = [[1, 2, 4],
         [2, 3, 4],
         [7, 8, 9]]
    a = np.roll(a, 1, axis=1)
    print(a)
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
