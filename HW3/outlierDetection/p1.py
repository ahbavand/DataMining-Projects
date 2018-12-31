print(__doc__)

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import pandas as pd
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


# Example settings
n_samples = 80
n_features = 5
repeat = 10
outliers_fraction = 0.25
rng = np.random.RandomState(42)



#data = scipy.io.loadmat("cardio.mat")

#for i in data:
#	if '__' not in i and 'readme' not in i:
#		np.savetxt(("filesforyou/"+i+".csv"),data[i],delimiter=',')





# define two outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=n_samples,
                                        contamination=outliers_fraction,
                                        random_state=rng)

}


# Fit the problem with varying cluster separation
X = pd.read_csv("salamX.csv")
Y = pd.read_csv("salamy.csv")
#for i in range(0,len(Y)):
#    print(i)
#    print(Y[i])
print(Y['label'][1])



# Fit the model
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    y_pred = clf.predict(X)
    print(len(y_pred),"salam")


    t=0
    for i in range(0,len(y_pred)):
        if((y_pred[i]==1 and(int)(Y['label'][i])==1) or (y_pred[i]==1 and(int)(Y['label'][i])==0)):
            print(y_pred[i])
            t=t+1
    print(t)





