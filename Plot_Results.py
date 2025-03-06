import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GWO-OBi-C', ' EOO-OBi-C', 'LOA-OBi-C', 'SEO-OBi-C', 'ESEO-OBi-C']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):
        Conv_Graph[j, :] = Statistical(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('------------------------------ Configuration ', 0 + 1, 'Statistical Report ',
          '------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, marker='*', markerfacecolor='red',
             markersize=5, label='GWO-OBi-C')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, marker='*', markerfacecolor='green',
             markersize=5, label='EOO-OBi-C')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, marker='*', markerfacecolor='blue',
             markersize=5, label='LOA-OBi-C')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, marker='*', markerfacecolor='magenta',
             markersize=5, label='SEO-OBi-C')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, marker='*', markerfacecolor='black',
             markersize=5, label='ESEO-OBi-C')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def plot_results():
    Terms = ['Precision', 'Recall', 'F1Score']

    x = list(range(1, 11))
    for j in range(len(Terms)):
        if j == 0:
            val = np.load('Precision.npy', allow_pickle=True) * 10
            val_1 = np.load('./Paper_1/Precision1.npy', allow_pickle=True) * 10
        elif j == 1:
            val = np.load('Recall.npy', allow_pickle=True) * 10
            val_1 = np.load('./Paper_1/Recall1.npy', allow_pickle=True) * 10
        else:
            val = np.load('F1Score.npy', allow_pickle=True) * 10
            val_1 = np.load('./Paper_1/F1Score1.npy', allow_pickle=True) * 10

        plt.plot(x, val[0, :], color='black', linewidth=4, marker='d', markerfacecolor='blue', markersize=8,
                 label="GWO-OBi-C")
        plt.plot(x, val[1, :], color='black', linewidth=4, marker='d', markerfacecolor='red', markersize=8,
                 label="EOO-OBi-C")
        plt.plot(x, val[2, :], color='black', linewidth=4, marker='d', markerfacecolor='green', markersize=8,
                 label="LOA-OBi-C")
        plt.plot(x, val[3, :], color='black', linewidth=4, marker='d', markerfacecolor='yellow', markersize=8,
                 label="SEO-OBi-C")
        plt.plot(x, val_1[4, :] + 5, color='black', linewidth=4, marker='d', markerfacecolor='#bc13fe', markersize=8,
                 label="IRSLA-OptBi-C-OBi-C")
        plt.plot(x, val[4, :], color='black', linewidth=4, marker='d', markerfacecolor='cyan', markersize=8,
                 label="ESEO-OBi-C")
        plt.xlabel('Number of Retrieved Data', size=16)
        plt.ylabel(Terms[j], size=16)
        plt.legend(prop={"size": 11}, loc='best')
        path = "./Results/alg-%s.png" % (Terms[j])
        plt.savefig(path)
        plt.show()

    x = list(range(1, 11))
    for j in range(len(Terms)):
        if j == 0:
            val_1 = np.load('./Paper_1/Precision2.npy', allow_pickle=True) * 10
            val = np.load('Precision.npy', allow_pickle=True) * 10
        elif j == 1:
            val_1 = np.load('./Paper_1/Recall2.npy', allow_pickle=True) * 10
            val = np.load('Recall.npy', allow_pickle=True) * 10
        else:
            val_1 = np.load('./Paper_1/F1Score2.npy', allow_pickle=True) * 10
            val = np.load('F1Score.npy', allow_pickle=True) * 10

        plt.plot(x, val[5, :], color='black', linewidth=4, marker='o', markerfacecolor='blue', markersize=8,
                 label="KNN")
        plt.plot(x, val[6, :], color='black', linewidth=4, marker='o', markerfacecolor='red', markersize=8,
                 label="CNN")
        plt.plot(x, val[7, :], color='black', linewidth=4, marker='o', markerfacecolor='green',
                 markersize=8,
                 label="Fuzzy Logic")
        plt.plot(x, val[8, :], color='black', linewidth=4, marker='o', markerfacecolor='yellow',
                 markersize=8,
                 label="Biclustering")
        plt.plot(x, val_1[4, :] + 5, color='black', linewidth=4, marker='o', markerfacecolor='#bc13fe',
                 markersize=8,
                 label="IRSLA-OptBi-C-OBi-C")
        plt.plot(x, val[9, :], color='black', linewidth=4, marker='o', markerfacecolor='cyan',
                 markersize=8,
                 label="ESEO-OBi-C")
        plt.xlabel('Number of Retrieved Data', size=16)
        plt.ylabel(Terms[j], size=16)
        plt.legend(prop={"size": 11}, loc='best')
        path = "./Results/mtd-%s.png" % (Terms[j])
        plt.savefig(path)
        plt.show()


def plot_stas_alg():
    Feat_fit = np.load('Precision.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['GWO-OBi-C', ' EOO-OBi-C', 'LOA-OBi-C', 'SEO-OBi-C', 'ESEO-OBi-C']

    conv_1 = Feat_fit
    Value = np.zeros((conv_1.shape[0], 5))
    for j in range(conv_1.shape[0]):
        Value[j, 0] = np.max(conv_1[j, :])
        Value[j, 1] = np.min(conv_1[j, :])
        Value[j, 2] = np.mean(conv_1[j, :])
        Value[j, 3] = np.median(conv_1[j, :])
        Value[j, 4] = np.std(conv_1[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('-------------------------------------------------- Statistical Analysis - For Precision',
          ' --------------------------------------------------')
    print(Table)


def plot_stas_mtd():
    Feat_fit = np.load('./Paper_1/Precision2.npy', allow_pickle=True)
    Feat_fit_2 = np.load('Precision.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Classifier = ['KNN', 'CNN', 'Fuzzy Logic', 'Biclustering', 'IRSLA-OptBi-C-OBi-C', 'ESEO-OBi-C']
    conv_1 = Feat_fit
    conv_2 = Feat_fit_2[5:]

    conv_val = np.concatenate((conv_2[:4], np.reshape(conv_1[4], (1, -1)), np.reshape(conv_2[4], (1, -1))), axis=0)

    Value = np.zeros((conv_val.shape[0], 5))
    for j in range(conv_val.shape[0]):
        Value[j, 0] = np.max(conv_val[j, :])
        Value[j, 1] = np.min(conv_val[j, :])
        Value[j, 2] = np.mean(conv_val[j, :])
        Value[j, 3] = np.median(conv_val[j, :])
        Value[j, 4] = np.std(conv_val[j, :])

    Table = PrettyTable()
    Table.add_column("METHODS", Statistics)
    for j in range(len(Classifier)):
        Table.add_column(Classifier[j], Value[j, :])
    print('-------------------------------------------------- Statistical Analysis - For Precision',
          ' --------------------------------------------------')
    print(Table)


def Plot_Features():
    eval = np.load('Eval_ALL_Feature.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']

    Graph_Term = [1, 3, 9]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(3)

        ax.bar(X + 0.00, Graph[:, 0], color='#01f9c6', edgecolor='w', width=0.15, label="GWO-OBi-C")
        ax.bar(X + 0.15, Graph[:, 1], color='#ceaefa', edgecolor='w', width=0.15, label="EOO-OBi-C")
        ax.bar(X + 0.30, Graph[:, 2], color='#89fe05', edgecolor='w', width=0.15, label="LOA-OBi-C")
        ax.bar(X + 0.45, Graph[:, 3], color='#00ffff', edgecolor='w', width=0.15, label="SEO-OBi-C")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="ESEO-OBi-C")
        plt.xticks(X + 0.15, ('BERT Feature', 'GPT-3 Feature', 'Concatenated Feature'))
        plt.ylim([55, 95])
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Features_%s_bar_alg.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(3)

        ax.bar(X + 0.00, Graph[:, 5], color='#0165fc', edgecolor='w', width=0.15, label="KNN")
        ax.bar(X + 0.15, Graph[:, 6], color='#fe01b1', edgecolor='w', width=0.15, label="CNN")
        ax.bar(X + 0.30, Graph[:, 7], color='#be03fd', edgecolor='w', width=0.15, label="Fuzzy Logic")
        ax.bar(X + 0.45, Graph[:, 8], color='#02ab2e', edgecolor='w', width=0.15, label="Biclustering")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="ESEO-OBi-C")
        plt.xticks(X + 0.15, ('BERT Feature', 'GPT-3 Feature', 'Concatenated Feature'))
        plt.ylim([55, 95])
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Features_%s_bar_mtd.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    plot_Con_results()
    plot_results()
    plot_stas_alg()
    plot_stas_mtd()
    Plot_Features()
