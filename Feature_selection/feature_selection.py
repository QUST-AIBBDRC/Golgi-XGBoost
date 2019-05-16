import numpy as np 
import pandas as pd
import ccm 
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
def mutual_mutual(data,label,k=200):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    mask=model_mutual.get_support(indices=True)
    return new_data,mask
def elasticNet(data,label,alpha =np.array([0.01]),l1_ratio=0.1):
    enet=ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.fit(data,label)
    mask_ = enet.coef_ 
    mask=np.nonzero(mask_)
    new_data = data[:,mask]
    return new_data,mask
def lassodimension(data,label,alpha=np.array([0.01,0.02,0.03])):
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data,label)
    mask = np.nonzero(x_lasso.coef_)
    new_data = data[:,mask]  
    return new_data,mask 
def ET(data,label):
    clf=ExtraTreesClassifier(n_estimators=200)
    clf=clf.fit(data,label)
    model=SelectFromModel(clf, prefit=True)
    X_new=model.transform(data)
    return X_new
def CCM_feature(X_res,y_res):
    epsilon = 0.1; num_features = 200; type_Y = 'binary'
    rank = ccm.ccm(X_res, y_res, num_features, type_Y, epsilon, iterations = 50, verbose = True)
    selected_feats = np.argsort(rank)[:200]
    X=X_res[:,selected_feats]
    return X
    
    

