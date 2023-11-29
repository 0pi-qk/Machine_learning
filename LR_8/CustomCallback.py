import tensorflow as tf
from matplotlib import pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, epochs, validation_data):
        super().__init__()
        self.epochs = epochs
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:
            accuracy = logs['accuracy']

            # Строим и сохраняем гистограмму точности
            plt.figure()
            plt.hist(accuracy, bins=10, density=True, alpha=0.75, color='b')  # density=True - нормализованная гистограмма
            plt.title(f'Accuracy Histogram - Epoch {epoch}')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
            plt.savefig(f'Epoch\{epoch}.png')
            plt.close()
