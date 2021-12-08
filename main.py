# python version
import sys
# scipy
import numpy as np
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

# url = "/Volumes/SD/Projects/PycharmProjects/pythonProject/AI_lab_01/drugDataTrain.csv"
train_path = '/Volumes/SD/Studia/Zima 2021/Sieci neuronowe L/Lista1/drugsCom_raw/drugsComTrain_raw.tsv'
test_path = '/Volumes/SD/Studia/Zima 2021/Sieci neuronowe L/Lista1/drugsCom_raw/drugsComTrain_raw.tsv'

names = ['drugName', 'condition', 'rating', 'date', 'usefulCount']

train_dataset = pandas.read_csv(train_path, delimiter='\t').dropna()
test_dataset = pandas.read_csv(test_path, delimiter='\t').dropna()

#zostawianie w tabeli tylko istotnych danych
# train_dataset = train_dataset[["drugName", "condition", " rating"]]
# test_dataset = test_dataset[["drugName", "condition", " rating"]]


# train_dataset = pandas.read_csv(train_path, delimiter='\t')
# print(len(train_dataset))
# train_dataset = train_dataset.dropna()


print(len(train_dataset))
train_dataset = train_dataset.head(100000) #zostawia tylko 500 rekordow
print(len(train_dataset))

# train_dataset = train_dataset.head(-500) #usuwa  500 rekordow

#lista lekow i chorob (string)
drugs_list = []
diseases_list = []


for index, row in train_dataset.iterrows():
    drug = row['drugName']
    disease = row['condition']
    if drug not in drugs_list: drugs_list.append(drug)
    if disease not in diseases_list: diseases_list.append(disease)


for index, row in train_dataset.iterrows():
    drug = row['drugName']
    disease = row['condition']
    if drug not in drugs_list: drugs_list.append(drug)
    if disease not in diseases_list: diseases_list.append(disease)

#encoder (nazwa string-> liczba indywidualna)

encoded_drugs_list = np.zeros((len(drugs_list), 1))
encoded_diseases_list = np.zeros((len(diseases_list), 1))
for i in range(0, len(drugs_list)):
    encoded_drugs_list[i] = i #ZAMIENIC NA LOSOWE LICZBY ?

for i in range(0, len(diseases_list)):
    encoded_diseases_list[i] = i

# replacing each string value with its corresponding index values
train_dataset = train_dataset.replace(diseases_list, encoded_diseases_list)
train_dataset = train_dataset.replace(drugs_list, encoded_drugs_list)


test_dataset = test_dataset.replace(diseases_list, encoded_diseases_list)
test_dataset = test_dataset.replace(drugs_list, encoded_drugs_list)

#usuwanie nadmiernych kolumn
# train_dataset = train_dataset[["drugName", "condition", " rating"]]
# test_dataset = test_dataset[["drugName", "condition", " rating"]]


# shape
# print(dataset.shape)

# head
print(train_dataset.head(10))

# description
#Print(dataset.describe())

# encoder


# class description
#print(dataset.groupby('condition').size())

# uzyc sklearn preprocessing  LabelEncodera


## box and whisker plots
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
