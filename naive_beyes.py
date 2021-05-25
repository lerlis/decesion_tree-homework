"""
@Time    : 2021/4/21
@Author  : Jupiter (朱比特)
@FileName: Navie_Bayes.py
@Blog    :https://blog.csdn.net/qq_1067857137
欢迎大家一起交流！！！
************************************************
naive_beyes.py是在上述代码博客代码的基础上修改而来
主要更改的部分是数据的输入，保持和decision_tree.py输入相同形式的数据，
即训练数据和测试数据的比例以及random.seed()均相同，
便于比较两种算法的性能
@修改时间   : 2021/5/25
@作者      : Le Xiangli
@文件名    : navie_beyes.py
************************************************
"""
import random
import numpy as np
from math import pi
from numpy import exp


def average_variance(per_dataset):
    """
    :param per_dataset:
    :return:[average]
    """
    average_num = sum(per_dataset) / len(per_dataset)
    value = 0
    for i in per_dataset:
        value = value + (i - average_num) ** 2
    variance_value = value / (len(per_dataset) - 1)
    return [average_num], variance_value


def data_reader(file):
    fopen = open(file, mode='r', encoding='utf-8')
    lines = fopen.readlines()
    data = []
    for line in lines:
        if line == '\n' or '':  # 去除空行
            continue
        else:
            perline = line.replace('\n', '').split(',')
            data.append(perline)
    fopen.close()
    return data


def data_filter(dataset):
    num = len(dataset)
    label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    numl = len(label)
    for i in range(num):
        for k in range(numl):
            if dataset[i][4] == label[k]:
                dataset[i][4] = float(k+1)
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
    return dataset


def data_shuffle(data, set_seed=1824):
    random.seed(set_seed)  # 1824
    random.shuffle(data)
    return data


def get_dataset():
    fpath = './dataset/iris.data'
    data = data_reader(fpath)
    fil_data = data_filter(data)
    # object = pd.read_excel(fpath)
    # object = shuffle(object)
    object = data_shuffle(fil_data, set_seed=1824)
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    # i1 = object.iloc[100:150, 0:5]
    # test_set = i1.values.tolist()  # 测试集
    test_set = object[105:150]
    train_set = object[0:105]
    # m1 = object.iloc[0:100, [0, 4]]
    # m2 = m1.values.tolist()
    m2 = []
    for line in train_set:
        m2.append([line[0], line[4]])
    for j in m2:
        if j[1] not in dict1:
            dict1[j[1]] = [j[0]]
        else:
            dict1[j[1]].append(j[0])
    print(dict1)
    # m1 = object.iloc[0:100, [1, 4]]
    # m2 = m1.values.tolist()
    m2 = []
    for line in train_set:
        m2.append([line[1], line[4]])
    for j in m2:
        if j[1] not in dict2:
            dict2[j[1]] = [j[0]]
        else:
            dict2[j[1]].append(j[0])
    # m1 = object.iloc[0:100, [2, 4]]
    # m2 = m1.values.tolist()
    m2 = []
    for line in train_set:
        m2.append([line[2], line[4]])
    for j in m2:
        if j[1] not in dict3:
            dict3[j[1]] = [j[0]]
        else:
            dict3[j[1]].append(j[0])
    # m1 = object.iloc[0:100, [3, 4]]
    # m2 = m1.values.tolist()
    m2 = []
    for line in train_set:
        m2.append([line[3], line[4]])
    for j in m2:
        if j[1] not in dict4:
            dict4[j[1]] = [j[0]]
        else:
            dict4[j[1]].append(j[0])

    return dict1, dict2, dict3, dict4, test_set


def constant_data(xi, average, variance):
    """
    此处是连续值，并且返回p(xi|c)
    :param xi: 指的是待预测的属性值
    :param average: 训练集该属性的平均值
    :param variance: 训练集该属性的方差
    :return:p(xi|c)
    """
    p_xi_c = (1 / (((2 * pi) ** 0.5) * variance)) * exp(-(xi - average) ** 2 / (2 * variance))
    return p_xi_c


def Naive_Bayesian(xi, average, variance):
    """
    朴素贝叶斯算法构建的过程
    :param xi:测试集的属性值
    :param average: 训练集该属性的平均值
    :param variance: 训练集该属性的方差
    :return: p(xi|c)
    """
    p_xi_c = (1 / (((2 * pi) ** 0.5) * variance)) * exp(-(xi - average) ** 2 / (2 * variance))
    p = constant_data(xi, average, variance)
    return p


if __name__ == '__main__':
    dict1, dict2, dict3, dict4, test_set = get_dataset()

    total_dict_var = []
    dict_var1 = []
    dict_var2 = []
    dict_var3 = []
    dict_var4 = []

    for i in range(1, 4):  # dict1:第一个属性对应1.0,2.0,3.0标签的列表集合
        average_1, variance_1 = average_variance(dict1[i])
        average_1.append((variance_1))
        dict_var1.append(average_1)
    print('接下来输出的是三个标签对应的第一个属性的[均值，方差]形式的列表集合')
    print(dict_var1)
    total_dict_var.append(dict_var1)
    # print(total_dict_var)
    for i in range(1, 4):  # dict2:第二个属性对应1.0,2.0,3.0标签的列表集合
        average_2, variance_2 = average_variance(dict2[i])
        average_2.append(variance_2)
        dict_var2.append(average_2)
    total_dict_var.append(dict_var2)
    for i in range(1, 4):  # dict3:第三个属性对应1.0,2.0,3.0标签的集合
        average_3, varince_3 = average_variance(dict3[i])
        average_3.append(varince_3)
        dict_var3.append(average_3)
    total_dict_var.append(dict_var3)
    for i in range(1, 4):  # dict4:第四个属性对应1.0,2.0,3.0标签的集合
        average_4, varince_4 = average_variance(dict4[i])
        average_4.append(varince_4)
        dict_var4.append(average_4)
    total_dict_var.append(dict_var4)
    print("接下来是total_dict_var(每一列存在三个标签，共四列")
    print(np.array(total_dict_var))  # 每一列存在三个标签 共存在4列
    # result = []    ->这个列表必须放在循环里面不然列表不会更新
    correct_num = 0  # 计数器,预测正确下面会自动+1
    for sample in test_set:
        result = []  # 此处是每个样例对应的 假设值(1.0,2.0,3.0)对应的相对概率值，之后会根据返回最大值的下标+1来进行预测的过程
        for i in range(0, 4):
            xi = sample[i]
            for j in range(0, 3):
                averaging = total_dict_var[i][j][0]
                variancing = total_dict_var[i][j][1]
                p_type = len(dict1[j + 1]) / (len(dict1[2]) + len(dict1[1]) + len(dict1[3]))  # 此处计算的是p(c)
                Navi1 = Naive_Bayesian(xi, averaging, variancing)
                result.append(Navi1)

        #   print(result)
        k = []
        o = []
        for j in range(0, 3):
            p_type = (len(dict1[j + 1])) / (len(dict1[2]) + len(dict1[1]) + len(dict1[3]))  # +1和+len(dict1[j+1])拉普拉斯修正
            o.append(p_type)
        # print(o)
        for i in range(0, 3):
            k.append(
                (result[i] * result[i + 3] * result[i + 6] + result[i + 9]) * o[i])  # 得到的列表里每个标签的属性间隔的下标是3 根据公式将其连乘
        # print(k)
        index_end = k.index(max(k)) + 1  # 列表里面排列的是三个标签值对应的p,排列顺序就是1.0,2.0,3.0,可以根据(index+1)来预测标签
        # print(index_end)
        if index_end == sample[4]:  # 比较预测预测值和实际值
            correct_num = correct_num + 1  # 同上面的计数器，正确则+1
    Accuracy1 = correct_num / len(test_set)  # 计算正确的概率
    print("Final Accuracy  ={}".format(Accuracy1))
