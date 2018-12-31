import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering



train = pd.read_csv('water-treatment.data.txt')

print(train.shape)
print(train.describe())
print(train.head())


def num_missing(x):
    return sum(x.isnull())


print(train.apply(num_missing, axis=0))

train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))

train = train.drop('G1', axis=1)

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train)
    print("k =", k)
    print(sklearn.metrics.silhouette_score(train, kmeans.labels_))

    # print(kmeans.labels_)
	
	
	
#train = pd.read_csv('./HTRU2/HTRU_2.csv')

#print(train.head())

#model = AgglomerativeClustering(2, linkage='ward')
#model.fit(train)

#Y = train['G8']
#X = train.drop('G8', axis=1)


#for index, linkage in enumerate(('complete', 'ward')):
#    plt.subplot(1, 3, index + 1)
#    model = AgglomerativeClustering(linkage=linkage, n_clusters=2)
#    predict = model.fit(X)
#    print(normalized_mutual_info_score(predict.labels_, Y))

