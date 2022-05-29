#从156个特征中选择代表性特征查看条形图
df6 = df3
df6['cluster'] = labels
columns = ['Cornea Back Astig', 'B Ele Th', 'ART max', 'Db', 'Max  Axis', 'Thinnest Locat Spot', 'WFA Back 9']
color = ['#79bd9a','#a8dba8','#3b8686']
plt.figure(figsize=(20,20))
for i in range(len(columns)):
    plt.subplot(3,3,i+1)
    plt.subplots_adjust(hspace=0.5)
    sns.set_style("whitegrid")
    sns.barplot(x="cluster", y=columns[i], data=df6, palette=color)
    plt.title("Barplot with cluster in "+columns[i])
plt.show()
