#聚类调参
i = []
y_silhouette_score = []
calinskiharabaz_score = []
for k in range(3,11):
    d = DBSCAN(eps=k, min_samples=2).fit(X_tsne)
    labels = d.labels_
    ss = sklearn.metrics.silhouette_score(X_tsne, labels)
    y_silhouette_score.append(ss)
    ass = sklearn.metrics.calinski_harabasz_score(X_tsne, labels)
    calinskiharabaz_score.append(ass)
    i.append(k)

plt.figure()
plt.plot(i,y_silhouette_score)  
plt.xlabel("dbscan-eps")  
plt.ylabel("silhouette_score")  
plt.title("silhouette_score")

plt.figure()
plt.plot(i,calinskiharabaz_score)  
plt.xlabel("dbscan-eps")  
plt.ylabel("calinskiharabaz_score")  
plt.title("calinskiharabaz_score")
plt.show()


#DBSCAN聚类
clustering = DBSCAN(eps=7, min_samples=2).fit(X_tsne)
labels = clustering.labels_
x0 = X_tsne[labels==0]
x1 = X_tsne[labels==1]
x2 = X_tsne[labels==2]

plt.figure(figsize=(10,8))
plt.scatter(x0[:,0], x0[:,1], c='#96ceb4', marker='o', label='label0')
plt.scatter(x1[:,0], x1[:,1], c='#d9534f', marker='o', label='label1')
plt.scatter(x2[:,0], x2[:,1], c='#ffad60', marker='o', label='label2')

plt.title('PCA_5_features_dbscan_method n_cluster=3')
plt.legend()
plt.show()

#为聚类样本加标签
df2['cluster'] = labels
#df2.to_csv("PCA_5_features_dbscan_method_n_cluster_3.csv")
