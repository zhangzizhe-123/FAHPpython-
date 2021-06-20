# _*_ coding:utf-8 _*_
# @Time : 2021/1/5 11:13
# @Author : zizhe
import numpy as np
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