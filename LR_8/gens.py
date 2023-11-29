import numpy as np


def gen_rect(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    w = np.random.randint(size // 10, size // 2)
    h = np.random.randint(size // 10, size // 2)
    img[x:x + w, y:y + h] = 1
    return img


def gen_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    for i in range(0, size):
        for j in range(0, size):
            if (i-x)**2 + (j-y)**2 <= r**2:
                img[i, j] = 1
    return img


def gen_empty_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    dr = np.random.randint(1, 10) + r
    for i in range(0, size):
        for j in range(0, size):
            if r**2 <= (i - x) ** 2 + (j - y) ** 2 <= dr ** 2:
                img[i, j] = 1
    return img


def gen_h_line(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size-10)
    y = np.random.randint(10, size-10)
    l = np.random.randint(size // 8, size // 2)
    w = 1
    img[x-w:x+w, y-l:y+l] = 1
    return img


def gen_v_line(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size - 10)
    y = np.random.randint(10, size - 10)
    l = np.random.randint(size // 8, size // 2)
    w = 1
    img[x - l:x + l, y - w:y + w] = 1
    return img


def gen_cross(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size - 10)
    y = np.random.randint(10, size - 10)
    l = np.random.randint(size // 8, size // 5)
    w = 1
    img[x-l:x+l, y-w:y+w] = 1
    img[x-w:x+w, y-l:y+l] = 1
    return img