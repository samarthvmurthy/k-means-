from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('kmeansdata.csv')
df1=pd.DataFrame(data)
f1=df1['Distance_Feature'].values
f2=df1['Speeding_Feature'].values
x=np.asarray(list(zip(f1,f2)))

plt.plot()
plt.xlim([0,100])
plt.ylim([0,50])
plt.title("Dataset")
plt.ylabel("speeding_feature")
plt.xlabel("distance_feature")
plt.scatter(f1,f2)
plt.show()

plt.plot()
colors=['b','g','r']
markers=['o','v','s']

kmeans_model=KMeans(n_clusters=3).fit(x)
plt.plot()

for i,l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i],f2[i],color=colors[l],marker=markers[l],ls='None')
plt.xlim([0,100])
plt.ylim([0,50])
plt.show()
