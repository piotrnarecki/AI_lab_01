# python version
import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# matplotlib
import pandas
# scikit-learn
import sklearn
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv(url, names=names)

# url = "/Users/piotrnarecki/Downloads/drugsCom_raw/drugsComTrain_raw.tsv"
# names = ['drugName', 'condition', 'review', 'rating', 'date','usefulCount']
# dataset = pandas.read_tsv(url, names=names)


import csv

# read tab-delimited file
with open('/Users/piotrnarecki/Downloads/drugsCom_raw/drugsComTrain_raw.tsv', 'r') as fin:
    cr = csv.reader(fin, delimiter='\t')
    filecontents = [line for line in cr]

# write comma-delimited file (comma is the default delimiter)
with open('drugDataTrain.csv', 'w') as fou:
    cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE, escapechar='\\')
    cw.writerows(filecontents)

# wyswietlanie

url = "/Volumes/SD/Projects/PycharmProjects/pythonProject/AI_lab_01/drugDataTrain.csv"
# names = ['drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
names = ['drugName', 'condition', 'review', 'rating']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# description
print(dataset.describe())

# class description
#print(dataset.groupby('class').size())

# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()
#
# # histograms
# dataset.hist()
# plt.show()
#
# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()
#
# # macierz do uczenia
# array = dataset.values
# X = array[:, 0:4]
# Y = array[:, 4]  # to co przewidujemy
# validation_size = 0.20  # z tego co mamy wykrajamy 20%
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                 random_state=seed)
# scoring = 'accuracy'
#
# # budowanie modelu
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))  # dodawanie algorytmow
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
#
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10)
#     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s :%f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
