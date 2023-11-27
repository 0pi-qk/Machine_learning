import numpy as np
import pandas as pd
import tensorflow as tf


# Поэлементные операции над тензорами
def oper_elem(x1, x2):
    return np.array([((x1[i] or x2[i]) and (x1[i] ^ (not (x2[i])))) for i in range(0, len(x1))])


# Операции реализованы с использованием операций над тензорами из NumPy
def oper_numpy(x1, x2):
    return np.logical_and(np.logical_or(x1, x2), np.logical_xor(x1, np.logical_not(x2)))

# dataset
x1 = [0, 1, 0, 1]
x2 = [0, 0, 1, 1]

ansElem = oper_elem(x1, x2)
ansNumpy = oper_numpy(x1, x2)

# Создание модели
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

predBefore = model.predict(np.array([x1, x2]).transpose())

# Переобучаем модель
history = model.fit(
    np.array([x1, x2]).transpose(),
    ansNumpy.transpose(),
    epochs=1000
)

predAfter = model.predict(np.array([x1, x2]).transpose())

# Вывод результатов
Result = pd.DataFrame(data={
    "x1": x1,
    "x2": x2,
    "oper_elem": ansElem,
    "oper_numpy": ansNumpy,
    "before": predBefore.tolist(),
    "after": predAfter.tolist()
})

Result.to_csv('Result.csv')
