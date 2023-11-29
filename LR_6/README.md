### Задание
Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.  
К каждому варианту прилагается код, который генерирует изображения.  
Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:
1. Тензор с изображениями ранга 3
2. Тензор с метками классов

Обратите внимание:
- Выборки не перемешаны, то есть наблюдения классов идут по порядку
- Классы характеризуются строковой меткой
- Выборка изначально не разбита на обучающую, контрольную и тестовую
- Скачивать необходимо оба файла. Подключать файл, который начинается с var (в нем и находится функция gen_data)

### Задание по варианту
Классификация изображений по количеству крестов на них. Может быть 1, 2 или 3

Файл 1 - gens.py:
```python
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
```

Файл 2 - var6.py:
```python
import gens
import numpy as np


def gen_k_cross(k, img_size=50):
    img = np.zeros([img_size, img_size])
    for i in range(k):
        img += gens.gen_cross(img_size)
    img[np.nonzero(img)] = 1
    return img


def gen_data(size=500, img_size=50):
    c1 = size // 3
    c2 = c1
    c3 = size - (c1 + c2)

    label_c1 = np.full([c1, 1], 'One')
    data_c1 = np.array([gen_k_cross(1, img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Two')
    data_c2 = np.array([gen_k_cross(2, img_size) for i in range(c2)])
    label_c3 = np.full([c3, 1], 'Three')
    data_c3 = np.array([gen_k_cross(3, img_size) for i in range(c3)])

    data = np.vstack((data_c1, data_c2, data_c3))
    label = np.vstack((label_c1, label_c2, label_c3))

    return data, label
```