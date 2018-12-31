import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn.cross_validation import train_test_split
from sklearn import tree, decomposition
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

train=pd.read_csv("X_train.csv")
train=train.drop("customerAttr_b",1)
train=train.drop("state",1)
train.drop("hour_a",1)
train.drop("total",1)
train["iszero"]=float('NaN')
train["iszero"][train["amount"]==0]=1
train["iszero"][train["amount"]>0]=0
train.drop("amount",1)



test=pd.read_csv("X_test.csv")
test=test.drop("customerAttr_b",1)
test=test.drop("state",1)
test.drop("hour_a",1)
test.drop("total",1)
test["iszero"]=float('NaN')
test["iszero"][test["amount"]==0]=1
test["iszero"][test["amount"]>0]=0
test.drop("amount",1)






pca = decomposition.PCA(n_components=3)


#print(train)
label=pd.read_csv("Y_train.csv")
dataa=[train,label]
data=pd.concat(dataa,axis=1)
#print(data)








number_records_fraud = len(data[data.fraud == 1])
fraud_indices = np.array(data[data.fraud == 1].index)
normal_indices = data[data.fraud == 0].index
#print(fraud_indices)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

#dar do khatte bala tedede barabar ba dade haye mosbat ra az dade haue manfi jada mikonim

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample_data = data.iloc[under_sample_indices,:]


X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'fraud']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'fraud']

#dar do khatte bala do ghesmate dade hadaf va baghie viZhegi ha ra za ham joda mikonim


X_train, X_test, y_train, y_test = train_test_split(data,label,test_size = 0.3, random_state = 0)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample ,y_undersample,test_size = 0.3 ,random_state = 0)


#nn = MLPClassifier(activation="relu", hidden_layer_sizes=(100,100,), alpha=0.00003)
#nn = nn.fit(X_train_undersample,y_train_undersample.values.ravel())
#scores = cross_val_score(nn, X_undersample, y_undersample.values.ravel(), cv = 10, scoring='accuracy')



rf = RandomForestClassifier(n_estimators = 100, warm_start=True, random_state=2, min_samples_leaf=2,max_depth=21)
rf = rf.fit(X_train_undersample,y_train_undersample.values.ravel())
scores = cross_val_score(rf, X_undersample, y_undersample.values.ravel(), cv = 10, scoring='accuracy')

my_prediction = rf.predict(test)

print(my_prediction)

so=pd.DataFrame(my_prediction)
so.to_csv("my_solution1.csv")











#lr = LogisticRegression(C =20, penalty = 'l1')
#lr.fit(X_train_undersample,y_train_undersample.values.ravel())
#scores = cross_val_score(lr, X_undersample, y_undersample.values.ravel(), cv = 10, scoring='accuracy')
#print(scores)


#dt = tree.DecisionTreeClassifier(random_state=1, max_depth=20)
#dt = dt.fit( X_undersample, y_undersample)
#scores = cross_val_score(dt, X_undersample, y_undersample.values.ravel(), cv = 10, scoring='recall')

#y_pred_undersample = lr.predict(X_test_undersample.values)


#lr = lr.fit( X_train_undersample, y_train_undersample.values.ravel())
#y_pred_undersample =lr.predict(X_test_undersample)
#cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
#print(cnf_matrix)



#lr = LogisticRegression(C =20, penalty = 'l1')
#dt = tree.DecisionTreeClassifier(random_state=1, max_depth=20)
#rf = RandomForestClassifier(n_estimators = 100, warm_start=True, random_state=2, min_samples_leaf=2,max_depth=21)
#combine=VotingClassifier(estimators=[("lr",lr),("dt",dt),("rg",rf)],voting='soft')
#combine.fit( X_undersample, y_undersample)
#scores = cross_val_score(combine, X_undersample, y_undersample.values.ravel(), cv = 10, scoring='recall')


avg=0
for i in range(0,10):
    avg+=scores[i]

avg/=10
print(avg)


#print(scores)
#y_pred_undersample = dt.predict(X_test_undersample.values)
#cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
#print(cnf_matrix)










#corr=train.corr()
#print(corr)
#count_classes = pd.value_counts(label['fraud'], sort = True).sort_index()
#count_classes.plot(kind = 'bar')
#print(count_classes)
#tr=train.ix[:, train.columns != 'amount']    #baraye joda kardane sotoon hai az dataframe

