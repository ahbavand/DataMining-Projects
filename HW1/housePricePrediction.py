import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif,f_regression,chi2
from sklearn import linear_model
from sklearn import ensemble
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn import preprocessing
from scipy.stats import skew




train = pd.read_csv("train2.csv")
test=pd.read_csv("test2.csv")

x= pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))


x=x.drop("MiscFeature",1)
x=x.drop("Fence",1)
x=x.drop("PoolQC",1)
x=x.drop("Alley",1)
x=x.drop("Street",1)
x=x.drop("Utilities",1)
x=x.drop("LandSlope",1)
x=x.drop("RoofMatl",1)
x=x.drop("Heating",1)
x=x.drop("FireplaceQu",1)
x=x.drop("LotFrontage",1)




#detecting columns that has nan value
def show_missing():
    missing = x.columns[x.isnull().any()].tolist()
    return missing


def cat_exploration(column):
    return x[column].value_counts()

def cat_imputation(column, value):
    x.loc[x[column].isnull(),column] = value












"""""

#impute cat
cat_imputation('MasVnrType', 'None')
cat_imputation('MasVnrArea', 0)

#impute electrical
cat_imputation('Electrical','SBrkr')

basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

for cols in basement_cols:
    if 'FinSF'not in cols:
        cat_imputation(cols,'None')


garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
for cols in garage_cols:
    if x[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)

x = x.fillna(x.mean())


"""

numeric_feats = x.dtypes[x.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

x[skewed_feats] = np.log1p(x[skewed_feats])


x = pd.get_dummies(x)
x = x.fillna(x.mean())

y=np.log(train["SalePrice"])


print(x[show_missing()].isnull().sum())
x_train=x[:train.shape[0]]
x_test = x[train.shape[0]:]

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)




#  linear model
#lin_model=linear_model.LinearRegression()
#print(rmse_cv(lin_model).mean())
#lin_model.fit(x_train,y)
#predict=lin_model.predict(x_test)




#gradient boosting

#boost_model=ensemble.GradientBoostingRegressor(n_estimators=50,max_depth=5, min_samples_split=88,learning_rate=0.3)
#print(rmse_cv(boost_model).mean())
#boost_model.fit(x_train,y)
#predict=boost_model.predict(x_test)











#lasso regression





#clf=linear_model.Lasso(alpha=0.5,max_iter=100,tol=0.001)
#print(rmse_cv(clf).mean())










#solution=pd.DataFrame({"SalePrice":predict,"id":test.Id, })
#solution.to_csv("pre3.csv", index = False)

#print(solution)





































#regr=linear_model.LinearRegression()
#scores=cross_val_score(regr,x,y,cv=10,scoring="mean_squared_error")
#regr.fit(x,y)
#rmse=0
#for i in range(0,10):
#    rmse+=sqrt(-scores[i])

#rmse/=10
#print(rmse)







#print(x[show_missing()].isnull().sum())
#print(train['SalePrice'].describe())
#print(cat_exploration('BsmtQual'))






#print(cat_exploration('MasVnrType'))



#print(x['LotFrontage'].corr(x['LotArea']))








#print(list(x))









