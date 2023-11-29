import var6
from CustomCallback import CustomCallback
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

# Генерация dataset
size = 1000
data = var6.gen_data(size=size, img_size=50)

x = np.array(data[0].reshape(len(data[0]), len(data[0][0]), len(data[0][0][0]), 1))

y = []
for i in range(len(data[1])):
    if data[1][i] == "One":
        y.append(0)
    elif data[1][i] == "Two":
        y.append(1)
    elif data[1][i] == "Three":
        y.append(2)

y = np.array(y)

# перемешиваем dataset
x, y = shuffle(x, y, random_state=0)
data = [x, y]

# разделение данных на тестовую и тренировочные выборки
div = round(size * 0.8)  # 0.8 - тренировочная, 0.2 - тестовая

train_feature = data[0][:div]  # срезы выборок
test_feature = data[0][div:]

train_target = data[1][:div]  # срезы целевых значений (target)
test_target = data[1][div:]

# Создание модели
model = tf.keras.Sequential([
    # Свёрточный слой
    tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu", input_shape=(50, 50, 1)),
    # Слой укрупнения признаков
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    # Слой для изменения размерности тензора (в одномерный)
    tf.keras.layers.Flatten(),
    # Полносвязный слой
    tf.keras.layers.Dense(128, activation="relu"),
    # Выходной слой
    tf.keras.layers.Dense(3, activation="softmax")
])

model.summary()

# компиляция модели
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Обучение модели
H = model.fit(
    train_feature, train_target,
    epochs=15,
    batch_size=256,
    validation_data=(test_feature, test_target),
    callbacks=CustomCallback(
        [0, 4, 5, 8, 13, 14],
        (test_feature, test_target)
    )
)
