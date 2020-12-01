__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Utility functions for the homework
'''

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def parse_data(feature_file, label_file):
    """
    :param feature_file: Tab delimited feature vector file
    :param label_file: class label
    :return: dataset as a pandas dataframe (features+label)
    """
    features = pd.read_csv(feature_file, sep="\t", header=None)
    labels = pd.read_csv(label_file, header=None)
    features['label'] = labels
    return features


def split_data(dataset):
    """
    Randomly choose 4,000 data points from the data files to form a training set, and use the remaining
    1,000 data points to form a test set. Make sure each digit has equal number of points in each set
    (i.e., the training set should have 400 0s, 400 1s, 400 2s, etc., and the test set should have 100 0s,
    100 1s, 100 2s, etc.)
    :param dataset: pandas datafrome (features+label)
    :return: None. Saves Train and Test datasets as CSV
    """
    # init empty dfs
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(0, 10):
        df = dataset.loc[dataset['label'] == i]
        train_split = df.sample(frac=0.8, random_state=200)
        test_split = df.drop(train_split.index)
        train_df = pd.concat([train_df, train_split])
        test_df = pd.concat([test_df, test_split])

    train_df.to_csv('dataset/MNIST_Train.csv', sep=',', index=False)
    test_df.to_csv('dataset/MNIST_Test.csv', sep=',', index=False)


def get_train_test(train_file, test_file):
    train = pd.read_csv(train_file, sep=",")
    y_train = train['label'].values.reshape(4000, 1)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    X_train = train.iloc[:, :-1].values

    test = pd.read_csv(test_file, sep=",")
    y_test = test['label'].values.reshape(1000, 1)
    y_test = mlb.fit_transform(y_test)
    X_test = test.iloc[:, :-1].values

    return X_train, y_train, X_test, y_test


def winning_count(X_test, w):
    winning_count_dict = {}
    for i, e in enumerate(range(0, 1000, 100)):
        winning_count_dict[i] = []
        for xi in X_test[e:e + 100, ]:
            winning_count_dict[i].append(winning_neuron(xi, w))

    return winning_count_dict


def reformat(winning_neuron_dict):
        winning_fraction_dict = {}
        for digit in winning_neuron_dict:
            winning_fraction_dict[digit] = np.zeros(144)
            for ind in winning_neuron_dict[digit]:
                winning_fraction_dict[digit][ind] += 1
            winning_fraction_dict[digit] = winning_fraction_dict[digit].reshape(12, 12)
            winning_fraction_dict[digit] = winning_fraction_dict[digit] / 100
        return winning_fraction_dict


def plot_winning_neurons(winning_fraction_dict):

    figs, ax = plt.subplots(2, 5)
    digit = 0
    for i in range(2):
        for j in range(5):
            ax[i][j].imshow(winning_fraction_dict[digit], cmap='hot')
            ax[i][j].axis('off')
            digit+=1
    plt.savefig('activity_map.pdf')


def plot_features(trained_w):
    reshaped_w = trained_w.reshape(12, 12, 784)
    figs, ax = plt.subplots(12, 12)
    for i in range(12):
        for j in range(12):
            ax[i][j].imshow(reshaped_w[i][j].reshape(28, 28).T, cmap='gray')
            ax[i][j].axis('off')
    plt.savefig('feature_map.pdf')


def plot_confusion_matrix(y_true, y_pred, file_name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    import seaborn as sns
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.savefig(file_name+'.pdf')
    plt.clf()
