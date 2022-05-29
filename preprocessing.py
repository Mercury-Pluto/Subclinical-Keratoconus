import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

#引入疑似圆锥数据集
data = pd.read_excel("data.xlsx", sheet_name = '疑似圆锥', header = 0)
#删掉年份、左右眼区分以及缺失过多的列
df = data.drop(["Unnamed: 1","Unnamed: 2","Eye","CTSP 10mm","PTI 10mm","front_sym","front_dis","back_sym","back_dis"],axis=1)
#删掉还有缺失值的行
df1 = df.dropna().reset_index()
df2 = df1.drop(["index"],axis=1)
df3 = df2.iloc[:,1:]

#PCA降维，保留80%信息量
pca = PCA(n_components=30)
pca.fit(df3)
X_pca = pca.transform(df3)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
df4 = pd.DataFrame(X_pca)

#画图显示前几个特征包含了多少信息量
np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_),marker='o')
plt.title('Scree Plot')
plt.show()

#使用降维数据，提取前几个pca特征，看二维空间效果
df5 = df4.iloc[:,:5]
plt.figure(figsize=(15,10))
iters = [250,270,400,1000]
for i in range(len(iters)):
    plt.subplot(2,2,i+1)
    tsne = TSNE(n_components=2, perplexity=35, learning_rate=200, n_iter=iters[i], init='random', random_state=10)
    X_tsne = tsne.fit_transform(df5)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], marker='.')
plt.show()

#查看前几个pca特征的相关性和区分度
sns.pairplot(df5,diag_kind='ked')
plt.show()
