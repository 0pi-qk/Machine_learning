import numpy as np
import random
import math
import matplotlib.pyplot as plt


def func(i):
    return (i % 16 + 1) / 16


def gen_sequence(seq_len=1000):
    seq = [math.cos(i / 10) * func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)), seq)
    plt.show()
