import numpy as np
import pandas as pd
from save_load import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learn Rate(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learn Rate(%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric)
    plt.legend(loc='upper right')
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)

def roc_curve_plot(y_test, y_pred):


    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('./Results/roc_curve 80.png', dpi=400)
    plt.show()






def densityplot(actual, predicted, learning_rate):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(actual, color='orange', label='Actual',  fill=True)
    sns.kdeplot(predicted, color='blue', label='Predicted',  fill=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title("density plot of Actual vs Predicted values")
    plt.legend()
    plt.savefig(f'Results/Density Plot Learning rate-{learning_rate}.png')
    plt.show()


def plot_res():
    # 70, 30 learning rate
    proposed_70 = load('proposed_70')
    dnn_70 = load('dnn_70')
    dbn_70 = load('dbn_70')
    rnn_70 = load('rnn_70')
    gru_70 = load('gru_70')

    data = {
        'DNN': dnn_70,
        'DBN': dbn_70,
        'RNN': rnn_70,
        'GRU': gru_70,
        'PROPOSED': proposed_70
    }

    ind = ['Accuracy', 'Precision','Recall', 'F-Measure', 'MCC', 'BSL']
    table = pd.DataFrame(data, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)
    save('table', table)
    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # 80, 20 learning rate

    proposed_80 = load('proposed_80')
    dnn_80 = load('dnn_80')
    dbn_80 = load('dbn_80')
    rnn_80 = load('rnn_80')
    gru_80 = load('gru_80')

    data1 = {
        'DNN': dnn_80,
        'DBN': dbn_80,
        'RNN': rnn_80,
        'GRU': gru_80,
        'PROPOSED': proposed_80
    }

    ind = ['Accuracy', 'Precision','Recall', 'F-Measure', 'MCC', 'BSL']
    table1 = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)
    save('table1', table1)
    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['DNN', 'DBN', 'RNN', 'GRU', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Recall', 'F-Measure', 'MCC', 'BSL']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i, :], metrices[1][i, :], metrices_plot[i])

    learn_data = [70, 80]
    for k in learn_data:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')
        densityplot(y_test, y_pred, k)

    y_test = load('y_test_70')
    y_pred = load('predicted_70')
    confu_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confu_matrix,
                                                display_labels=['Non Attack', 'Attack'])
    fig, ax = plt.subplots(figsize=(12, 8))
    cm_display.plot(ax=ax)

    plt.savefig('./Results/confusion_matrix 70 .png', dpi=400)
    plt.show()

    y_test = load('y_test_80')
    y_pred = load('predicted_80')
    confu_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confu_matrix,
                                                display_labels=['Non Attack', 'Attack'])
    fig, ax = plt.subplots(figsize=(12, 8))
    cm_display.plot(ax=ax)

    plt.savefig('./Results/confusion_matrix 80 .png', dpi=400)
    plt.show()

    roc_curve_plot(y_test, y_pred)