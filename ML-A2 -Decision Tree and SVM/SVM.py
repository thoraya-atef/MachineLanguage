import numpy as np
import pandas as pd


class Support_Vector_Machine:
    def __init__(self, learning_rate=0.0001, lambda_=0.01,
                 iterations=1000):  # if learning rate =0.0001 ->95.86206896551724 %
        self.learning_rate = learning_rate  # learning rate so small =0.000001  ->95.17241379310344 %
        self.lambda_ = lambda_  # learning rate big = 0.01  ->95.17241379310344 %
        self.iterations = iterations
        self.W = None
        self.Bais = None

    def Fit_method(self, X, Y):
        New_Y = np.where(Y <= 0, -1, 1)
        numberOfrows, numberOffeatures = X.shape
        self.W = np.zeros(numberOffeatures)
        self.Bais = 0
        for i in range(self.iterations):
            for current_index, current_sample in enumerate(X):
                Condition = New_Y[current_index] * (
                            np.dot(self.W, current_sample) - self.Bais) >= 1  # if y.f(x)>=1 condition=true
                if Condition:
                    dj_dw = 2 * self.lambda_ * self.W  # derivation for W
                    self.W -= self.learning_rate * dj_dw
                else:
                    dj_dw = 2 * self.lambda_ * self.W - np.dot(current_sample, New_Y[current_index])  # derivation for W
                    self.W -= self.learning_rate * dj_dw
                    self.Bais -= self.learning_rate * New_Y[current_index]

    def Predict_method(self, X):
        linear = np.dot(X, self.W) - self.Bais
        return np.sign(linear)


def main():
    # import data from Excel sheet
    Data = pd.read_csv('heart.csv')
    # training
    Data_training = Data.head(212)
    X = Data_training.drop(['target', 'cp', 'slope', 'thal', 'restecg', 'ca'], axis='columns')
    Y = Data_training['target']
    X = np.array(X)
    Y = np.array(Y)

    # testing
    Data_testing = Data.tail(150)
    X_test = Data_testing.drop(['target', 'cp', 'slope', 'thal', 'restecg', 'ca'], axis='columns')
    Y_test = Data_testing['target']
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # age,chol,thalach,trestbps    <<<important features>>>
    # drop>>> ca,restecg
    object = Support_Vector_Machine()  # object from svm class
    object.Fit_method(X, Y)
    print("vector of w =", object.W)
    print("Bais =", object.Bais)

    Y_predict = np.zeros(len(Y_test))
    for index in range(len(Y_test)):
        flag = object.Predict_method(X_test)[index] <= 0
        if flag:
            Y_predict[index] = 0
        else:
            Y_predict[index] = 1
    # print(Y_predict)

    # accuracy
    # (correctly predicted values / test set size)
    accuracy = (np.sum(np.equal(Y_test, Y_predict)) / len(Y_test)) * 100
    print("Accuracy =", accuracy, "%")


if __name__ == '__main__':
    main()
