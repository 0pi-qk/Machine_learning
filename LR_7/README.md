### Задание

Необходимо построить рекуррентную нейронную сеть, которая будет прогнозировать значение некоторого периодического сигнала.

Простой пример рекуррентной сети, которая предсказывает значение зашумленной синусоиды. Модель из примера не является наилучшей, а лишь демонстрирует пример построения сети со слоями GRU и LSTM.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np 
import random
import math
import matplotlib.pyplot as plt


def gen_sequence(seq_len = 1000):
    seq = [math.sin(i/10) + random.normalvariate(0, 0.09) for i in range(seq_len)]
    return np.array(seq)


def gen_data_from_sequence(seq_len = 1006, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)

data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(32,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
model.add(layers.LSTM(32,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.2))
model.add(layers.GRU(32,input_shape=(None,1),recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data,train_res,epochs=50,validation_data=(val_data, val_res))

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)),loss)
plt.plot(range(len(val_loss)),val_loss)
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length,predicted_res)
plt.plot(pred_length,test_res)
plt.show()
```

К каждому варианту предоставляется код, который генерирует последовательность. Для выполнения задания необходимо:
1. Преобразовать последовательность в датасет, который можно подавать на вход нейронной сети (можно использовать функцию gen_data_from_sequence из примера)
2. Разбить датасет на обучающую, контрольную и тестовую выборку
3. Построить и обучить модель
4. Построить график последовательности, предсказанной на тестовой выборке (пример построения также есть в примере). Данный график необходимо также добавить в pr

Также, в файлах с кодом вариантов есть функция draw_sequence, которая позволяет нарисовать часть последовательности

### Задание по варианту
Код для генерации последовательности:
```python
import numpy as np 
import random
import math
import matplotlib.pyplot as plt

def func(i):
    return (i % 16 + 1) / 16

def gen_sequence(seq_len = 1000):
    seq = [math.cos(i/10) * func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

draw_sequence()
```