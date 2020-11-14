import pandas as pd
import matplotlib.pyplot as plt
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

def sklearn_knn(X_train, y_train, X_test, y_test, ks):
    mces = []
    for k in ks:
        neigh = KNeighborsClassifier(n_neighbors = k)
        neigh.fit(X_train, y_train)
        pred = neigh.predict(X_test)
        mce = calc_mce(pred, y_test)
        mces.append(mce)
    return mces

def part_4(X_train, y_train, X_test, y_test):
    ks = [1, 5, 10]
    mces = sklearn_knn(X_train, y_train, X_test, y_test, ks)
    ks = [1/k for k in ks]
    plt.plot(ks, mces)
    plt.scatter(ks, mces)
    plt.xlabel('1/k')
    plt.ylabel('misclassification error')
    plt.title('misclasification error vs 1/k')
    plt.show()

def part_5(X_train, y_train, X_test, y_test):
    neigh = KNearestNeighbors()
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    mce = calc_mce(pred, y_test)
    print('The misclassification rate for the naive implementation of KNN is:', mce)

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

    part_4(X_train, y_train, X_test, y_test)
    
    part_5(X_train, y_train, X_test, y_test)
    
