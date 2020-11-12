import pandas as pd
import matplotlib.pyplot as plt
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
test_dataset = pd.read_csv(train_dataset_path,
 names=["x1", "x2", "class"],
 delimiter=" ")
X_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, -1].values

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.show()
