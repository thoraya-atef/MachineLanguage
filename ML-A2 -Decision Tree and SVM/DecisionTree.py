import array as arr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Node:
    def __init__(self, featidx=None, left=None, right=None, InfGin=None, value=None):
        self.feature_idx = featidx
        self.left = left  # left child
        self.right = right  # right child
        self.InformationGain = InfGin  # calc by function
        # for leaf node "political party"
        self.value = value


class DecisionTree:
    def __init__(self, X=None, Y=None):
        self.classes = np.unique(Y)  # 2 republican,democrat
        self.num_samples = X.shape[0]  # X.shape get num or row and num of columns we get num of rows
        self.num_features = X.shape[1]  # X.shape get num or row and num of columns we get num of columns 16
        self.root = None

    def Entropy(self, y):
        count_Rep = 0
        count_Dem = 0
        y = np.array(y)
        for i in range(len(y)):
            if y[i] == 'republican':
                count_Rep = count_Rep + 1
            elif y[i] == 'democrat':
                count_Dem = count_Dem + 1
        Prob_Rep = count_Rep / (count_Dem + count_Rep)
        Prob_Dem = count_Dem / (count_Dem + count_Rep)
        entropy = (-1 * Prob_Rep * np.log2(Prob_Rep)) - (Prob_Dem * np.log2(Prob_Dem))  # H(s)
        return entropy

    def INFO_Gain(self, parent, left_child, right_child):
        ratio_l = len(left_child) / len(parent)
        ratio_r = len(right_child) / len(parent)
        gain = parent.entropy - ((ratio_l * left_child.entropy) + (ratio_r * right_child.entropy))
        return gain

    def Split(self, dataset, idx, label):
        left = np.array([dataset.iloc[row][:] for row in range(len(dataset)) if dataset.iloc[row][idx] == label])
        right = np.array([dataset.iloc[row][:] for row in range(len(dataset)) if dataset.iloc[row][idx] != label])
        return left, right

    def Best_Split(self, dataset):
        maxInfoGain = -float("inf")  # -infinity
        for idx in range(0, self.num_features):
            branch_l, branch_r = self.Split(dataset, idx, 'y')
            if len(branch_l) > 0 and len(branch_r) > 0:
                parent = dataset.iloc[:, -1]  # last column is class
                left_child = pd.DataFrame(branch_l[:, -1], columns=['Class'])
                left_child = left_child.iloc[:, -1]
                right_child = pd.DataFrame(branch_r[:, -1], columns=['Class'])
                right_child = right_child.iloc[:, -1]
                parent.entropy = self.Entropy(parent)
                left_child.entropy = self.Entropy(left_child)
                right_child.entropy = self.Entropy(right_child)
                info_gain = self.INFO_Gain(parent, left_child, right_child)
                if info_gain > maxInfoGain:
                    index = list()
                    index.append(idx)
                    feature_idx = pd.DataFrame(index, columns=['feature_idx'])
                    left = pd.DataFrame(branch_l, columns=['handicapped-infants', 'water-project-cost-sharing',
                                                           'adoption-of-the-budget-resolution',
                                                           'physician-fee-freeze', 'el-salvador-aid',
                                                           'religious-groups-in-schools', 'anti-satellite-test-ban',
                                                           'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
                                                           'synfuels-corporation-cutback',
                                                           'education-spending',
                                                           'superfund-right-to-sue', 'crime', 'duty-free-exports',
                                                           'export-administration-act-south-africa', 'Class'])
                    right = pd.DataFrame(branch_r, columns=['handicapped-infants', 'water-project-cost-sharing',
                                                            'adoption-of-the-budget-resolution',
                                                            'physician-fee-freeze', 'el-salvador-aid',
                                                            'religious-groups-in-schools',
                                                            'anti-satellite-test-ban',
                                                            'aid-to-nicaraguan-contras', 'mx-missile',
                                                            'immigration', 'synfuels-corporation-cutback',
                                                            'education-spending',
                                                            'superfund-right-to-sue', 'crime', 'duty-free-exports',
                                                            'export-administration-act-south-africa', 'Class'])
                    gain = list()
                    gain.append(info_gain)
                    InformationGain = pd.DataFrame(gain, columns=["InformationGain"])
                    best_split = feature_idx
                    best_split.loc[:, 'InformationGain'] = InformationGain
                    best_split.feature_idx = idx
                    best_split.left = left
                    best_split.right = right
                    best_split.InformationGain = InformationGain
                    maxInfoGain = info_gain
            if maxInfoGain == -float("inf"):
                best_split = list()
                best_split.append(-1)
                best_split = pd.DataFrame(best_split, columns=["best_split"])
                return best_split
        return best_split

    def build_tree(self, dataset, currentDepth=0):
        y = dataset.iloc[:, -1]
        # split until num samples =2
        if self.num_samples >= 2:
            # to find the best split
            best_split = self.Best_Split(dataset)
            # check if information gain is positive
            if int(best_split.iloc[0][0]) != -1:
                # print(best_split.iloc[0][1])
                if best_split.iloc[0][1] > 0:
                    # left
                    leftSubtree = self.build_tree(best_split.left, currentDepth + 1)
                    # right
                    rightSubtree = self.build_tree(best_split.right, currentDepth + 1)
                    # return decision node
                    return Node(int(best_split.iloc[0][0]), leftSubtree,
                                rightSubtree, best_split.iloc[0][1])
        # compute leaf node
        leaf_value = self.CalculatePoliticalParty(y)
        return Node(value=leaf_value)

    # compute leaf node
    def CalculatePoliticalParty(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def printTree(self, tree=None, indent=" "):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
            return tree.value

        else:
            print("Issues _num " + str(tree.feature_idx))
            print("%sleft:" % (indent), end="")
            self.printTree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.printTree(tree.right, indent + indent)

    def TreeSize(self, node):
        if node is None:
            return 0
        else:
            return self.TreeSize(node.left) + 1 + self.TreeSize(node.right)

    def fit(self, X, Y):
        dataset = X
        dataset = dataset.assign(Class=Y)
        self.root = self.build_tree(dataset)
        return self.root

    def predict(self, X):
        Y_predicate = [self.prediction(x, self.root) for x in X]
        return Y_predicate

    def prediction(self, x, tree):
        if tree.value != None:
            # print(tree.value)
            return tree.value
        feature_val = x[tree.feature_idx]
        if feature_val == 'y':
            return self.prediction(x, tree.left)
        else:
            return self.prediction(x, tree.right)


def accuracy(given_y, pred_y):
    accuracy = (np.sum(given_y == pred_y) / len(given_y)) * 100
    return accuracy


if __name__ == '__main__':
    path = "house-votes-84.data.txt"  # path of data set
    names = ['Class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
             'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
             'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
             'education-spending',
             'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    dataset = pd.read_csv(path, sep=",", header=None, names=names)
    # print(dataset)
    for j in range(1, len(dataset.columns)):
        count_yes_re = 0
        count_no_re = 0
        count_yes_de = 0
        count_no_de = 0
        for i in range(len(dataset)):
            # print(j, i, dataset.columns[j], dataset.iloc[i][j])
            if dataset.iloc[i][0] == "republican":
                if dataset.iloc[i][j] == 'y':
                    count_yes_re = count_yes_re + 1
                elif dataset.iloc[i][j] == 'n':
                    count_no_re = count_no_re + 1
            else:
                if dataset.iloc[i][j] == 'y':
                    count_yes_de = count_yes_de + 1
                elif dataset.iloc[i][j] == 'n':
                    count_no_de = count_no_de + 1
        for i in range(len(dataset)):
            if dataset.iloc[i][j] == '?':
                if dataset.iloc[i][0] == "republican":
                    if count_yes_re >= count_no_re:
                        dataset.iloc[i][j] = 'y'
                    else:
                        dataset.iloc[i][j] = 'n'
                else:
                    if count_yes_de >= count_no_de:
                        dataset.iloc[i][j] = 'y'
                    else:
                        dataset.iloc[i][j] = 'n'
    for i in range(5):
        df = dataset.sample(frac=0.25)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        print("Tree Size = ", tree.TreeSize(root))
        Y_predict = tree.predict(np.array(x_test))
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy", i + 1, ": ", accuracy(np.array(y_test), Y_predict))
        print("----------------------------------------------------------------------------------")
    accuracy1 = arr.array('f')
    depth1 = arr.array('i')
    print("--------------------------- random seed = 100 ------------------------------")
    trainSize = 0.3
    while trainSize <= 0.7:
        df = dataset.sample(frac=trainSize, random_state=100)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        d = tree.TreeSize(root)
        depth1.append(d)
        print("Tree Size = ", d)
        Y_predict = tree.predict(np.array(x_test))
        a = accuracy(np.array(y_test), Y_predict)
        accuracy1.append(a)
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy of Train Size", trainSize * 100, "% : ", a)
        print("----------------------------------------------------------------------------------")
        trainSize += 0.1

    print("--------------------------- random seed = 170 ------------------------------")
    accuracy2 = arr.array('f')
    depth2 = arr.array('i')
    trainSize = 0.3
    while trainSize <= 0.7:
        df = dataset.sample(frac=trainSize, random_state=170)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        d = tree.TreeSize(root)
        depth2.append(d)
        print("Tree Size = ", d)
        Y_predict = tree.predict(np.array(x_test))
        a = accuracy(np.array(y_test), Y_predict)
        accuracy2.append(a)
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy of Train Size", trainSize * 100, "% : ", a)
        print("----------------------------------------------------------------------------------")
        trainSize += 0.1

    print("--------------------------- random seed = 225 ------------------------------")
    accuracy3 = arr.array('f')
    depth3 = arr.array('i')
    trainSize = 0.3
    while trainSize <= 0.7:
        df = dataset.sample(frac=trainSize, random_state=225)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        d = tree.TreeSize(root)
        depth3.append(d)
        print("Tree Size= ", d)
        Y_predict = tree.predict(np.array(x_test))
        a = accuracy(np.array(y_test), Y_predict)
        accuracy3.append(a)
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy of Train Size", trainSize * 100, "% : ", a)
        print("----------------------------------------------------------------------------------")
        trainSize += 0.1

    print("--------------------------- random seed = 270 --------------------------------")
    accuracy4 = arr.array('f')
    depth4 = arr.array('i')
    trainSize = 0.3
    while trainSize <= 0.7:
        df = dataset.sample(frac=trainSize, random_state=270)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        d = tree.TreeSize(root)
        depth4.append(d)
        print("Tree Size = ", d)
        Y_predict = tree.predict(np.array(x_test))
        a = accuracy(np.array(y_test), Y_predict)
        accuracy4.append(a)
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy of Train Size", trainSize * 100, "% : ", a)
        print("----------------------------------------------------------------------------------")
        trainSize += 0.1
    print("--------------------------- random seed = 300 ------------------------------------")
    accuracy5 = arr.array('f')
    depth5 = arr.array('i')
    trainSize = 0.3
    while trainSize <= 0.7:
        df = dataset.sample(frac=trainSize, random_state=300)
        # print(df)
        X_train = df.loc[:, dataset.columns != 'Class']
        Y_train = df['Class']
        x_test = dataset.loc[:, dataset.columns != 'Class'].drop(X_train.index)
        y_test = dataset['Class'].drop(Y_train.index)
        tree = DecisionTree(X_train, Y_train)
        root = tree.fit(X_train, Y_train)
        tree.printTree()
        d = tree.TreeSize(root)
        depth5.append(d)
        print("Tree Size= ", d)
        Y_predict = tree.predict(np.array(x_test))
        a = accuracy(np.array(y_test), Y_predict)
        accuracy5.append(a)
        # print("predicated", len(Y_predict),Y_predict)
        # print("test",len(y_test),np.array(y_test))
        print("Accuracy of Train Size", trainSize * 100, "% : ", a)
        print("----------------------------------------------------------------------------------")
        trainSize += 0.1

    trainSize = 0.3
    i = 0
    meanacc = arr.array('f')
    meandep = arr.array('f')
    train = arr.array('i', [30, 40, 50, 60, 70])
    # plt.scatter(range(0,100), range(0,100))
    while trainSize <= 0.7:
        print("For training set size", trainSize * 100, "% :")
        acc_per_training = arr.array('f')
        acc_per_training.append(accuracy1[i])
        acc_per_training.append(accuracy2[i])
        acc_per_training.append(accuracy3[i])
        acc_per_training.append(accuracy4[i])
        acc_per_training.append(accuracy5[i])
        print("\t Min Accuracy", min(acc_per_training))
        print("\t Max Accuracy", max(acc_per_training))
        meanacc.append((sum(acc_per_training) / len(acc_per_training)))
        print("\t Mean Accuracy", (sum(acc_per_training) / len(acc_per_training)))
        depth_per_training = arr.array('f')
        depth_per_training.append(depth1[i])
        depth_per_training.append(depth2[i])
        depth_per_training.append(depth3[i])
        depth_per_training.append(depth4[i])
        depth_per_training.append(depth5[i])
        print("\t Min Tree Size", min(depth_per_training))
        print("\t Max Tree Size", max(depth_per_training))
        meandep.append((sum(depth_per_training) / len(depth_per_training)))
        print("\t Mean Tree Size", (sum(depth_per_training) / len(depth_per_training)))
        del acc_per_training[:]
        del depth_per_training[:]
        print("----------------------------------------")
        trainSize += 0.1
        i += 1
    plt.plot(train, meanacc, color='red')
    plt.show()
    plt.plot(train, meandep, color='blue')
    plt.show()
