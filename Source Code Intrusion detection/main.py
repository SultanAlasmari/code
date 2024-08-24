'''
pip install pandas
pip install missingpy
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install openpyxl
'''

import os
os.makedirs('Data Visualization', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)

from datagen import datagen
from save_load import load, save
from Detection import Ensemble_Net, DNN, DBN, RNN, gru
from ARAVOA import ARAVOA
from Objective_function import fit_func_80, fit_func_70
import matplotlib.pyplot as plt
from plotResult import plot_res


def full_analysis():
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    learning_data = [(x_train_70, y_train_70, x_test_70, y_test_70, fit_func_70),
                     (x_train_80, y_train_80, x_test_80, y_test_80, fit_func_80)]

    i = 70
    for train_test_set in learning_data:
        x_train, y_train, x_test, y_test, fit_func = train_test_set

        # optimize epoch, batch size for IoTGuard EnsembleNet-based IoT Attack Detection(proposed): model

        lb = [50, 32]
        ub = [100, 64]

        problem_dict1 = {
            "fit_func": fit_func,
            "lb": lb,
            "ub": ub,
            "minmax": "min",
        }
        epoch = 1000
        pop_size = 50
        model = ARAVOA(epoch, pop_size)
        best_position, best_fitness = model.solve(problem_dict1)

        optimal_epoch = int(best_position[0])
        optimal_batch_size = int(best_position[1])

        predicted, metrics, history = Ensemble_Net(x_train, y_train, x_test, y_test, optimal_epoch, optimal_batch_size)

        plt.figure(figsize=(10, 4))
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([-0.01, 1.01])
        plt.legend(loc='lower right')
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([-0.01, 0.51])
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'Results/Accuracy Loss Graph - {i}.png')
        plt.show()

        pred, met = DNN(x_train, y_train, x_test, y_test)
        save(f'dnn_{i}', met)

        pred, met = DBN(x_train, y_train, x_test, y_test)
        save(f'dbn_{i}', met)

        pred, met = RNN(x_train, y_train, x_test, y_test)
        save(f'rnn_{i}', met)

        pred, met = gru(x_train, y_train, x_test, y_test)
        save(f'gru_{i}', met)

        i = 80


a = 0
if a == 1:
    full_analysis()


plot_res()
plt.show()