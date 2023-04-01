# -*-coding:utf-8-*-
"""
@Time：2022/7/17 7:28
@Auth：黄祎
@File：03_snownlp_comparison
@IDE：PyCharm
@Type：
@Remark：
@History：
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
print(X)
Z = linkage(X, 'ward')
f = fcluster(Z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
