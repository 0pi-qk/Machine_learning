import var6
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

# Генерация dataset
size = 1000
data = var6.gen_data(size=size, img_size=50)

# 1 крест
plt.imshow(data[0][size // 3 - 1], interpolation='nearest')
plt.show()

# 2 креста
plt.imshow(data[0][size // 3 * 2 - 1], interpolation='nearest')
plt.show()

# 3 креста
plt.imshow(data[0][size - 1], interpolation='nearest')
plt.show()

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
history = model.fit(
    train_feature, train_target,
    epochs=15,
    batch_size=256,
    validation_data=(test_feature, test_target)
)

# Получение ошибки и точности
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# График ошибки
plt.plot(epochs, loss, 'bo', color="red", label='Training loss')
plt.plot(epochs, val_loss, 'bo', color="blue", label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# График точности
plt.clf()
plt.plot(epochs, acc, 'bo', color="red", label='Training acc')
plt.plot(epochs, val_acc, 'bo', color="blue", label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# работа с данными для построения гистограмм
pred_probability = model.predict(data[0])

pred = pd.DataFrame(pred_probability).idxmax(axis=1)

target_true = data[1]

pred_true = []
pred_false = [0, 0, 0]
for i in range(len(pred)):
    if pred[i] == target_true[i]:
        pred_true.append(pred[i])
    else:
        pred_false[target_true[i]] += 1

# Гистограмма предсказаний
legend = [
    "Исходные",
    "Предсказанные",
    "Предсказанные верно"
]
plt.hist(
    [target_true, pred, pred_true],
    histtype='bar', label=legend)
plt.title("Гистограмма предсказаний")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("Количество")
plt.xlabel("Группы")
plt.show()

# Гистограмма неверных ответов
labels = ['1', '2', '3']
plt.bar(labels, pred_false)

plt.show()
