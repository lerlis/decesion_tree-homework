import treePlotter
import numpy as np
import random
import collections
import math
import operator

# 全局变量，用于标注结点和叶子节点
leaf = 0
node = 0


def data_reader(file):
    """
    读取数据文件，返回原始数据
    :param file:
    :return:
    """
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


def data_filter(dataset, label):
    """
    将数据集中的标签转换为数字，并将字符串转换为浮点数
    :param dataset:
    :param label:
    :return:
    """
    num = len(dataset)
    numl = len(label)
    for i in range(num):
        for k in range(numl):
            if dataset[i][4] == label[k]:
                dataset[i][4] = k
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
    return dataset


def data_split(data, ratio, set_seed=1824):
    """
    将过滤后的数据划分为训练集和测试集，并将属性和标签分开
    :param data:
    :param ratio: 训练集和测试集的比例
    :param set_seed: 便于复现结果
    :return:
    """
    num = len(data)
    num_train = int(num * ratio)
    random.seed(set_seed)  # 1824,124,12
    random.shuffle(data)
    train = data[0:num_train]
    test = data[num_train:]
    x_train = [i[0:4] for i in train]
    y_train = [i[4] for i in train]
    x_test = [i[0:4] for i in test]
    y_test = [i[4] for i in test]
    return x_train, y_train, x_test, y_test


def calinfoEntropy(x_train, y_train):
    """
    计算某一个属性的信息熵
    :param x_train:
    :param y_train:
    :return:
    """
    assert len(x_train) == len(y_train)
    num = len(x_train)
    sEntropy = 0.0
    num_label = collections.defaultdict(int)  # 便于构建一个字典，可以填入原本不存在的key
    for i in range(num):
        current_label = y_train[i]
        num_label[current_label] += 1
    for key in num_label:
        p = float(num_label[key]) / num
        sEntropy = sEntropy - p * math.log(p, 2)  # 计算信息熵
    return sEntropy


def choosebestattribute(x_train, y_train):
    """
    选择当前节点下拥有最好分类效果的属性以及划分点
    :param x_train:
    :param y_train:
    :return:
    """
    num_attributes = len(x_train[0])
    base_entropy = calinfoEntropy(x_train, y_train)
    best_info_gain = 0
    best_feature = -1
    best_mid = 0
    acc_info_gain = []
    acc_mid = []
    for i in range(num_attributes):
        infoGain, mid = seriescalbestIG(x_train, y_train, i, base_entropy)
        acc_info_gain.append(infoGain)
        acc_mid.append(mid)
        if infoGain > best_info_gain:  # 找到最优的Information Gain所属的类别和划分条件
            best_mid = mid
            best_feature = i
            best_info_gain = infoGain
    return best_feature, best_mid, acc_info_gain, acc_mid


def seriescalbestIG(x_train, y_train, col, base_entropy):
    """
    针对某一特定的属性计算连续条件下的Information Gain
    :param x_train:
    :param y_train:
    :param col:
    :param base_entropy:
    :return:
    """
    maxinfoGain = 0
    bestmid = -1
    feature_list = [line[col] for line in x_train]
    class_list = [y for y in y_train]
    dict_list = dict(zip(feature_list, class_list))  # 获得该属性的所有长度
    sorted_list = sorted(dict_list.items(), key=operator.itemgetter(0))  # 对长度排序
    num_sorted_list = len(sorted_list)
    mid_list = []
    for k in range(num_sorted_list - 1):
        mid_list.append(round(float(sorted_list[k][0] + sorted_list[k + 1][0]) / 2.0, 3))  # 计算所有的中间值，3位小数
    for mid in mid_list:
        greater_list, y_list_g, less_equal_list, y_list_l = splitseriesdata(x_train, y_train, col, mid)
        attribu_entropy = float(len(greater_list)) / len(x_train) * calinfoEntropy(greater_list, y_list_g) + float(
            len(less_equal_list)) / len(x_train) * calinfoEntropy(less_equal_list, y_list_l)
        info_gain = base_entropy - attribu_entropy  # 计算Information Gain
        if info_gain > maxinfoGain:
            bestmid = mid
            maxinfoGain = info_gain
    return maxinfoGain, bestmid


def splitseriesdata(x_train, y_train, col, mid):
    """
    将原数据集x_train按照中间值mid划分为全部大于和小于等于两个数据集，
    标签集y_train同步
    :param x_train:
    :param y_train:
    :param col:
    :param mid:
    :return:
    """
    greater_list = []
    y_list_g = []
    less_equal_list = []
    y_list_l = []
    assert len(x_train) == len(y_train)
    num = len(x_train)
    for i in range(num):
        if x_train[i][col] > mid:  # 比中间值大，计入greater_list
            greater_list.append(x_train[i])
            y_list_g.append(y_train[i])
        else:
            less_equal_list.append(x_train[i])
            y_list_l.append(y_train[i])
    return greater_list, y_list_g, less_equal_list, y_list_l


def thres_select(num_of_label):
    """
    统计只剩两种分类标签时较多的数据标签和较少的数据标签并返回
    :param num_of_label:
    :return:
    """
    count = []
    feature_save = []
    for key in num_of_label:
        a = num_of_label[key]
        count.append(a)
        feature_save.append(key)
    if count[0] > count[1]:
        y_show = feature_save[0]
        y_not_show = feature_save[1]
    else:
        y_show = feature_save[1]
        y_not_show = feature_save[0]
    return y_show, y_not_show


def tree_creator(x_train, y_train, Attributes, label, threshold):
    """
    递归的方式生成树，并采用字典的形式进行储存
    :param x_train:
    :param y_train:
    :param Attributes:
    :param label:
    :return:
    """
    global node, leaf

    num_of_label = collections.defaultdict(int)
    for i in range(len(y_train)):
        current_label = y_train[i]
        num_of_label[current_label] += 1
    if len(num_of_label) == 2:
        for key in num_of_label:
            a = int(num_of_label[key])
            if a <= threshold:
                y_show, y_not_show = thres_select(num_of_label)
                leaf += 1
                print('LEAF:第%s个叶节点' % leaf, '下包含的数据类别和个数为:', y_show, y_not_show, len(y_train)
              , '\n', y_train)
                return '%s:%s' % (leaf, label[y_show])
            else:
                continue

    # 当标签中只剩下一种标签时，说明分类可以结束了，终止节点(叶子节点)直接存入一个字符串
    if y_train.count(y_train[0]) == len(y_train):
        leaf += 1
        print('LEAF:第%s个叶节点' % leaf, '下包含的数据类别和个数为:', label[y_train[0]], len(y_train)
              , '\n', y_train)
        return '%s:%s' % (leaf, label[y_train[0]])

    node += 1
    # 选择当前节点或者所剩数据下的最优属性以及划分
    feature, mid, acc_Info_gain, acc_mid_t = choosebestattribute(x_train, y_train)
    print('NODE:第%s个结点,以维度%s划分' % (node, feature), '即为', Attributes[feature], '中间值为：', mid)
    print('各个Attribute的IG值为', acc_Info_gain, '各个Attribute的划分点为', acc_mid_t)
    # 生成决策树
    best_feature_label = str(node) + ':' + str(Attributes[feature]) + '=' + str(mid)

    myTree = {best_feature_label: {}}  # 用于递归，存储决策树

    gr_list, y_g, le_eq_list, y_l = splitseriesdata(x_train, y_train, feature, mid)

    subattri = Attributes[:]
    sublabel = label[:]

    subTree = tree_creator(gr_list, y_g, subattri, sublabel, threshold)
    myTree[best_feature_label]['great'] = subTree  # 如果不是叶子节点则存入一个字典

    subTree = tree_creator(le_eq_list, y_l, subattri, sublabel, threshold)
    myTree[best_feature_label]['less'] = subTree

    return myTree


def get_true_false(tree, x_test, y_test, label, attribute):
    """
    计算预测值，并和真实值做一个对应关系的统计
    :param tree:
    :param x_test:
    :param y_test:
    :param label:
    :param attribute:
    :return:
    """
    matrix = {label[0]: {label[0]: 0, label[1]: 0, label[2]: 0},
              label[1]: {label[0]: 0, label[1]: 0, label[2]: 0},
              label[2]: {label[0]: 0, label[1]: 0, label[2]: 0}}
    assert len(x_test) == len(y_test)
    for i in range(len(y_test)):
        predict = find_classification(tree, x_test[i], y_test[i], attribute)
        actual = label[y_test[i]]
        matrix[actual][predict] += 1
    return matrix


def find_classification(Tree, x, y, attribute):
    """
    根据测试集中的数据和已有的决策树，计算预测标签，并返回
    :param Tree:
    :param x:
    :param y:
    :param attribute:
    :return:
    """
    first_label = list(Tree.keys())[0]
    sub_dictionary = Tree[first_label]
    seperate1 = first_label.index(':')
    seperate2 = first_label.index('=')
    feature = attribute.index(first_label[seperate1 + 1: seperate2])
    mid = float(first_label[seperate2 + 1:])
    for key in sub_dictionary.keys():
        if x[feature] > mid:
            if type(sub_dictionary['great']).__name__ == 'dict':
                predict_label = find_classification(sub_dictionary['great'], x, y, attribute)
            else:
                predict_label = sub_dictionary['great']
                seperate3 = predict_label.index(':')
                predict_label = predict_label[seperate3 + 1:]
            return predict_label
        else:
            if type(sub_dictionary['less']).__name__ == 'dict':
                predict_label = find_classification(sub_dictionary['less'], x, y, attribute)
            else:
                predict_label = sub_dictionary['less']
                seperate3 = predict_label.index(':')
                predict_label = predict_label[seperate3 + 1:]
            return predict_label


def get_accuracy(mat, label):
    """
    计算准确率
    :param mat:
    :param label:
    :return:
    """
    num = len(mat)
    accuracy_each = []
    accuracy = 0
    for i in range(num):
        correct_num = mat[label[i]][label[i]]
        all_num = mat[label[i]][label[0]] + mat[label[i]][label[1]] + mat[label[i]][label[2]]
        accuracy_each.append(float(correct_num) / all_num)
    for acc in accuracy_each:
        accuracy += acc
    return accuracy_each, float(accuracy) / 3


if __name__ == "__main__":
    data_origin = data_reader('./dataset/iris.data')
    # print(data_origin)
    label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    Attributes = ['sepal length', 'sepal width', 'petal length', 'petal width']
    fil_data = data_filter(data_origin[:], label)
    # print(fil_data)
    x_train, y_train, x_test, y_test = data_split(fil_data, ratio=0.6, set_seed=1824)
    # print('1:', x_train)
    # print('2:', y_train)
    # print('3:', x_test)
    # print('4:', y_test)
    ShannonEnt = calinfoEntropy(x_train, y_train)
    # print('训练集信息熵：', ShannonEnt)
    feature, mid, acc_IG, acc_t = choosebestattribute(x_train, y_train)
    # print('第1次分类维度：', feature, '即为', Attributes[feature],'中间值划分：', mid)
    # print('第1次划分的IG值为', acc_IG, '第1次各属性的划分点为', acc_t)
    print("***************************开始训练**************************")
    myTree = tree_creator(x_train, y_train, Attributes, label, threshold=0)
    print(myTree)
    treePlotter.createPlot(myTree)
    print("***************************测试部分**************************")
    matrix = get_true_false(myTree, x_test, y_test, label, Attributes)
    print('正确和错误统计矩阵：\n', matrix)
    accuracy_each, accuracy = get_accuracy(matrix, label)
    print('准确率：', accuracy)
    print('每种类别的准确率：', accuracy_each)
