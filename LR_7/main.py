import tensorflow as tf
from var6 import *

# Генерируем сигнал
seq = gen_sequence()


# Преобразование сигнала в форму пригодную для обучения
def gen_data_from_sequence(seq, lookback=10):
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future


# Разбиение dataset на тестовую, обучающую и валидационные выборки
data, res = gen_data_from_sequence(seq)

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]

# Построение модели
model = tf.keras.Sequential([
    # Управляемый рекуррентный блок Gated Recurrent Units (GRU)
    tf.keras.layers.GRU(64, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True),
    # Долгая краткосрочная память (рекуррентный слой)
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2),
    # Управляемый рекуррентный блок
    tf.keras.layers.GRU(64, input_shape=(None, 1), recurrent_dropout=0.2),
    # Выходной слой
    tf.keras.layers.Dense(1),
])

model.summary()

# Компиляция
model.compile(
    optimizer='nadam',
    loss='mse',
)

# Обучение
history = model.fit(
    train_data,
    train_res,
    epochs=50,
    validation_data=(val_data, val_res)
)

# Получение ошибки и точности
loss = history.history['loss']
val_loss = history.history['val_loss']
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
predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res, color="red", label='Предсказанные значения')
plt.plot(pred_length, test_res, color="blue", label='Контрольные значения')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
