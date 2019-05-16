import numpy as np 
import pandas as pd
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import scale
from pyHSICLasso import HSICLasso
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import utils.tools as utils
#X_random=np.random.random((100,200))
#X =X_random
#y_random=np.random.randint(2,size=100)
#y=y_random
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
def mutual_mutual(data,label,k=100):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    mask=model_mutual.get_support(indices=True)
    return new_data,mask

start = time.time()
data=pd.read_csv('train_smote.csv',header=0)

data=np.array(data)
row=data.shape[0] 
column=data.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)#shuffle the index
index=np.array(index)
data_=data[index,:]
shu=data_[:,2:]
label=data_[:,1]
y=label
y[y==2]=0

#X_=scale(shu)
X_=shu
new_X,mask=mutual_mutual(shu,y,k=200)
X=new_X
#X_shuffle,y=get_shuffle(new_X,y_,random_state=1)
#X_initial=X_shuffle
loo = LeaveOneOut()
sepscores = []
y_score=np.ones((1,2))*0.5
y_class=np.ones((1,1))*0.5       
for train, test in loo.split(X):
    cv_clf = xgb.XGBClassifier(max_depth=10, learning_rate=0.01,
                 n_estimators=50, silent=True,
                objective="binary:logistic", booster='gbtree'
                )
    #cv_clf = BalancedRandomForestClassifier(n_estimators=500,max_depth=15, random_state=0)
    #cv_clf = SVC(probability=True)#using support vector machine
    X_train=X[train]
    y_train=y[train] 
    X_test=X[test]
    y_test=y[test]
    y_sparse=utils.to_categorical(y)
    y_train_sparse=utils.to_categorical(y_train)
    y_test_sparse=utils.to_categorical(y_test)
    hist=cv_clf.fit(X_train, y_train)
    y_predict_score=cv_clf.predict_proba(X_test) 
    y_predict_class= utils.categorical_probas_to_classes(y_predict_score)
    y_score=np.vstack((y_score,y_predict_score))
    y_class=np.vstack((y_class,y_predict_class))
    cv_clf=[]
y_class=y_class[1:]
y_score=y_score[1:]
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
roc_auc = auc(fpr, tpr)
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y)
result=[acc,precision,npv,sensitivity,specificity,mcc,roc_auc]
row=y_score.shape[0]
#column=data.shape[1]
y_sparse=utils.to_categorical(y)
yscore_sum = pd.DataFrame(data=y_score)
yscore_sum.to_csv('yscore_xgb_MI.csv')
ytest_sum = pd.DataFrame(data=y_sparse)
ytest_sum.to_csv('ytest_xgb_MI.csv')
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
auc_score=result[6]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='XGB ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('result_MI.csv')
interval = (time.time() - start)
print("Time used:",interval)
print("acc=%.2f%% " % (result[0]*100))
print("sensitivity=%.2f%% " % (result[3]*100))
print("specificity=%.2f%% " % (result[4]*100))
print("mcc=%.2f%%" % (result[5]*100))
print("auc=%.2f%%" % (result[6]*100))


 