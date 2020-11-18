import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from naive_knn import KNearestNeighbors

#plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
#plt.show()

def calc_mce(pred, gold):
    counter = 0
    for p, g in zip(pred, gold):
        if p != g:
            counter += 1
    return counter / len(pred)

def sklearn_knn(X_train, y_train, X_test, y_test, k):
    neigh = KNeighborsClassifier(n_neighbors = k) #, weights = lambda x: np.where(x == 0, 1, 1 / (x ** 2))) # this was used as sanity check for part 7
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    mce = calc_mce(pred, y_test)
    print('For k = %d the misclassification rate of sklearn KNN is: %.3f' % (k, mce))
    return mce

def part_4(X_train, y_train, X_test, y_test):
    ks = [1, 5, 10]
    mces = []
    for k in ks:
        mces.append(sklearn_knn(X_train, y_train, X_test, y_test, k))
    ks = [1/k for k in ks]
    plt.plot(ks, mces)
    plt.scatter(ks, mces)
    plt.xlabel('1/k')
    plt.ylabel('misclassification error')
    plt.title('misclasification error vs 1/k')
    plt.show()

kinds = ['naive', 'weighted']

def part_5(X_train, y_train, X_test, y_test, k, kind = 'naive'):
    assert kind in kinds
    neigh = KNearestNeighbors(n_neighbors = k)
    neigh.fit(X_train, y_train)
    if kind == 'naive':
        pred = neigh.predict(X_test)
    else:
        pred = neigh.predict_weighted(X_test)
    mce = calc_mce(pred, y_test)
    print('For k = %d the misclassification rate for %s KNN is: %.3f' % (k, kind, mce))
    return mce

def part_7(X_train, y_train, X_test, y_test):
    ks = [1, 5, 10]
    mces = []
    for k in ks:
        mces.append(part_5(X_train, y_train, X_test, y_test, k, kind = 'weighted'))
    ks = [1/k for k in ks]
    plt.plot(ks, mces)
    plt.scatter(ks, mces)
    plt.xlabel('1/k')
    plt.ylabel('misclassification error')
    plt.title('misclasification error vs 1/k')
    plt.show()

if __name__ == "__main__":
    dataset_dir = "https://raw.githubusercontent.com/probml/pmtkdata/master/knnClassify3c/"
    train_dataset_path = dataset_dir + "knnClassify3cTrain.txt"
    test_dataset_path = dataset_dir + "knnClassify3cTest.txt"
    # Train dataset
    train_dataset = pd.read_csv(train_dataset_path,
        names=["x1", "x2", "class"],
        delimiter=" ")
    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values
    # Test dataset
    test_dataset = pd.read_csv(test_dataset_path,
        names=["x1", "x2", "class"],
        delimiter=" ")
    X_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values

    #part_4(X_train, y_train, X_test, y_test)
    
    #part_5(X_train, y_train, X_test, y_test, 1)

    #part 6 is only the implementation in the naive_knn.py file

    part_7(X_train, y_train, X_test, y_test)
    
