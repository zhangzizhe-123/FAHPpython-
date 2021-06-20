# _*_ coding:utf-8 _*_
# @Time : 2020/12/22 21:37
# @Author : zizhe
import numpy as np
import pandas as pd
import warnings
import xlrd


class AHP:
    def __init__(self, criteria, factors):
        self.RI = (0, 0, 0.58, 0.90, 1.12, 1.24)  # RI：平均随机一致性指标
        self.criteria = criteria  # 准则
        self.factors = factors  # 因素
        self.num_criteria = criteria.shape[0]
        self.num_factors = factors[0].shape[0]

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, '不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反对称矩阵')
        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)  # 计算矩阵的特征值，特征向量
        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n]
        return max_eigen, CR, eigen

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        print('准则层: 最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))

        max_eigen_list, CR_list, eigen_list = [], [], []
        k = 1
        for i in self.factors:
            max_eigen, CR, factors_eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(factors_eigen)
            print('准则{}因素层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(k, max_eigen, CR, '' if CR < 0.1 else '不'))
            print('因素层权重={}\n'.format(factors_eigen))
            k = k + 1
        return criteria_eigen, eigen_list


def fuzzey_eval(criteria, eigen):
    # 量化评语(优秀，良好，一般，差)
    score = [1.0, 0.8, 0.6, 0.25]
    df = get_DataFrameExcel()
    print('单因素模糊综合评价：{}\n'.format(df))
    # 把单因素评价数据，拆解到3个准则中
    v1 = df.iloc[0:3, :].values
    v2 = df.iloc[3:6, :].values
    v3 = df.iloc[6:10, :].values
    vv = [v1, v2, v3]

    val = []
    num = len(eigen)
    for i in range(num):
        v = np.dot(np.array(eigen[i]), vv[i])
        print('准则{}, 矩阵积为：{}'.format(i + 1, v))
        val.append(v)
    # 目标层
    obj = np.dot(criteria, np.array(val))
    print('目标层模糊综合评价：{}\n'.format(obj))
    # 综合评分
    eval = np.dot(np.array(obj), np.array(score).T)
    print('综合评价：{}'.format(eval*100))

    return


def get_DataFrameExcel():
    excel_name = input('输入文件名')
    df = pd.read_excel(excel_name)
    return df


def main():
    # 准则重要性矩阵
    criteria = np.array([[1, 1/5, 1/3], [5, 1, 2], [3, 1/2, 1]])
    # 对每个准则，方案优劣排序
    b1 = np.array([[1, 1/3, 1/5], [3, 1, 1/4], [5, 4, 1]])
    b2 = np.array([[1, 1/3, 1/7], [3, 1, 1/5], [7, 5, 1]])
    b3 = np.array([[1, 1/3, 1/7, 1/9], [3, 1, 1/3, 1/7], [7, 3, 1, 1/5], [9, 7, 5, 1]])

    b = [b1, b2, b3]
    a, c = AHP(criteria, b).run()
    fuzzey_eval(a, c)


if __name__ == '__main__':
    main()
