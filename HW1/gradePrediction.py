import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score



train=pd.read_csv("train.csv",header=None,names=["g1","g2","g3","g4","g5","g6","g7"])
test=pd.read_csv("test.csv",header=None,names=["g1","g2","g3","g4","g5","g6"])
t_test=test








train['g1'][train['g1'] == 0] =np.NaN
train['g2'][train['g2'] == 0] =np.NaN
train['g3'][train['g3'] == 0] =np.NaN
train['g4'][train['g4'] == 0] =np.NaN
train['g5'][train['g5'] == 0] =np.NaN
train['g6'][train['g6'] == 0] =np.NaN
test['g1'][train['g1'] == 0] =np.NaN
test['g2'][train['g2'] == 0] =np.NaN
test['g3'][train['g3'] == 0] =np.NaN
test['g4'][train['g4'] == 0] =np.NaN
test['g5'][train['g5'] == 0] =np.NaN
test['g6'][train['g6'] == 0] =np.NaN



def show_missing():
    missing = train.columns[train.isnull().any()].tolist()
    return missing


print(train[show_missing()].isnull().sum())








train['g1'].fillna(train['g1'].mean(), inplace=True)
train['g2'].fillna(train['g2'].mean(), inplace=True)
train['g3'].fillna(train['g3'].mean(), inplace=True)
train['g4'].fillna(train['g4'].mean(), inplace=True)
train['g5'].fillna(train['g5'].mean(), inplace=True)
train['g6'].fillna(train['g6'].mean(), inplace=True)




test['g1'].fillna(train['g1'].mean(), inplace=True)
test['g2'].fillna(train['g2'].mean(), inplace=True)
test['g3'].fillna(train['g3'].mean(), inplace=True)
test['g4'].fillna(train['g4'].mean(), inplace=True)
test['g5'].fillna(train['g5'].mean(), inplace=True)
test['g6'].fillna(train['g6'].mean(), inplace=True)


def imputation(dataset):
#fill NaN by Normal distribution
    for i in range (1, 7):
        dataset["g{}".format(i)][dataset["g{}".format(i)]==0]=None
        mean = dataset["g{}".format(i)].mean()
        std =np.std(dataset['g{}'.format(i)])
        dataset['g{}'.format(i)].fillna(np.random.normal(mean,std), inplace=True)
        j = 0
        #when we use nor distribution it can cuase some grades become out of range
        #(0,20)
        for j in range (0,len(dataset)):
            if(dataset['g{}'.format(i)][j]>20):
                dataset['g{}'.format(i)][j]=20
            if(dataset['g{}'.format(i)][j]<0):
                dataset['g{}'.format(i)][j]=0






#x=train[["g1","g2","g3","g4","g5","g6"]][0:200]
#target=train[["g7"]][0:200]
#x_test=train[["g1","g2","g3","g4","g5","g6"]][200:]
#target_test=train[["g7"]][200:]

#imputation(train)
X = train[['g1', 'g2','g4','g5','g6']].values
Y = train['g7'].values
#alpha=[20,15,10,7,5,3,1,0.7,0.5,0.3,0.1,0.05,0.01,0.005,0.001,0.0005]
#max_iter=[100000,50000,30000,10000,5000,3000,1000,2000,1000,500,200,100,150,80,60,30,10,5]
#tol=[1000,100,1,10,0.1,0.01,0.001,0.0001,0.000001,0.00001]

#min=10
#a=[0,]
#clf=linear_model.Lasso(alpha=0.7,tol=0.0001)
#scores = np.sqrt(-cross_val_score(clf, X, Y, cv=10, scoring="mean_squared_error"))
#print(scores.mean())

""""
for i in range(len(alpha)):
    for j in range(len(max_iter)):
        for k in range(len((tol))):

            clf=linear_model.Lasso(alpha=alpha[i],max_iter=max_iter[j],tol=tol[k])
            scores=cross_val_score(clf,X,Y,cv=10,scoring="mean_squared_error")
            print(scores)
            rmse=0

            for i in range(0,10):
                rmse+=sqrt(-scores[i])

            rmse/=10

            if(rmse<min):
                min=rmse
            a.append(rmse)
            print(rmse)

print(min)
"""

#regr=linear_model.LinearRegression()
#scores=cross_val_score(regr,X,Y,cv=10,scoring="mean_squared_error")
#rmse=0
#for i in range(0,10):
#    rmse+=sqrt(-scores[i])

#rmse/=10
#print(rmse)
#regr.fit(train[['g1','g2','g4','g5','g6']],train['g7'])

#test["g7"]=regr.predict(test[['g1','g2','g4','g5','g6']])
#print(test)





#ridge

#model_ridge = Ridge(alpha=10000000)
#scores=cross_val_score(model_ridge,X,Y,cv=10,scoring="mean_squared_error")
#rmse=0
#for i in range(0,10):
#    rmse+=sqrt(-scores[i])

#rmse/=10
#print(rmse)












#g2 = train['g6']
#g7 = train['g7']
#plt.plot(g2, g7, 'ro')
#plt.show()
#print(train.corr(method='pearson'))






boost_model=ensemble.GradientBoostingRegressor(n_estimators=17,max_depth=2, min_samples_split=4,learning_rate=0.33)

#scores=cross_val_score(boost_model,X,Y,cv=10,scoring="mean_squared_error")
#rmse=0
#for i in range(0,10):
#    rmse+=sqrt(-scores[i])

#rmse/=10
#print(rmse)
boost_model.fit(train[['g1','g2','g3','g4','g6']],train['g7'])

t_test['g7']=boost_model.predict(test[['g1','g2','g3','g4','g6']])
t_test.to_csv("p1_submission.csv",index=False)





