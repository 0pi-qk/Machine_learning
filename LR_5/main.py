import random
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import load_model
import matplotlib.pyplot as plt


# Получение х в заданном диапазоне
def gen_rand_X():
    return round(random.uniform(0, 10), 2)


# Получение е в заданном диапазоне
def gen_rand_E():
    return round(random.uniform(0, 0.3), 2)

# генерация dataset
size = 1000

feature = [[] for _ in range(6)]
target = []

for _ in range(size):
    x = gen_rand_X()
    e = gen_rand_E()

    feature[0].append(np.cos(x) + e)
    feature[1].append(-x + e)
    target.append(np.sin(x) * x + e)
    feature[2].append(np.sqrt(np.abs(x)) + e)
    feature[3].append(x ** 2 + e)
    feature[4].append(-np.abs(x) + 4)
    feature[5].append(x - (x ** 2) / 5 + e)

data = pd.DataFrame({
    "Признак 1": feature[0],
    "Признак 2": feature[1],
    "Признак 4": feature[2],
    "Признак 5": feature[3],
    "Признак 6": feature[4],
    "Признак 7": feature[5],
    "Цель": target
})

data.head()
data.to_csv("dataset.csv")

# Выгружаем dataset из файла
data = pd.read_csv("dataset.csv", index_col=0)
data.head()

# Делим на тренировочную и тестовые выборки
div = round(size * 0.8)

train_feature = data.iloc[:div, 0:6]
test_feature = data.iloc[div:, 0:6]

train_target = data.iloc[:div, 6:7]
test_target = data.iloc[div:, 6:7]

# Входной слой
main_input = tf.keras.Input(shape=(6,), dtype="float32", name="main_input")

# Слой кодирования
coder_output = tf.keras.layers.Dense(8, activation="relu", name="coder_output")(main_input)

# Скрытые слои
x = tf.keras.layers.Dense(16, activation="relu")(coder_output)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
x = tf.keras.layers.Dense(8, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation="relu")(x)

# Выходной слой регрессионной модели
decoder_output = tf.keras.layers.Dense(1, activation="sigmoid", name="decoder_output")(x)

# Скрытые слои
x = tf.keras.layers.Dense(16, activation="relu")(coder_output)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
x = tf.keras.layers.Dense(8, activation="relu")(x)

# Выходной слой регрессионной модели
regression_output = tf.keras.layers.Dense(1, name="regression_output")(x)

model = tf.keras.Model(inputs=[main_input], outputs=[regression_output, decoder_output])

model.compile(
    optimizer="rmsprop",
    loss="mean_squared_logarithmic_error",
    metrics=['accuracy'],
    loss_weights=[1., 1.]
)

# Обучение модели
H = model.fit(
    train_feature,
    train_target,
    epochs=100,
    batch_size=16,
    validation_data=(
        test_feature,
        test_target
    )
)

# получение ошибки и точности обучения
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(1, len(loss) + 1)

# посторонние графика
plt.plot(epochs, loss, 'bo', color="red", label='Training loss')
plt.plot(epochs, val_loss, 'bo', color="blue", label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# сохраняем модель
model.save('model.h5')
del model
model = load_model('model.h5')

# Выгружаем модель из файла
pred = model.predict(test_feature)

# Записываем результаты в файл
out = pd.DataFrame()
out["Исходное"] = test_target.Цель.tolist()
out["regression"] = [item for sublist in pred[0].tolist() for item in sublist]
out.head()

out.to_csv("result.csv")
