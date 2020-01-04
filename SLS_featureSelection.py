#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:29:13 2019

@author: Hanieh Marvi Khorasani 
"""
## this code is an optimization on state-of-the-art feature selection methods
#including SVMRFE, mRMR and HSICLASSO
#first irrelevant feature are recognized and removed from dataset and then feature selection
# will be applied on the reduced dataset. Repeated stratified cross validation is used for model creation
# and validation
### input : name of dataset , name of feature selection method, name of classification method
### outputs: validation metrics (Classification accuracy, Area under the curve, recall, specificity,
# area under precision-recall curve) & number of optimal features

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import RFE
import pymrmr
from sklearn.svm import SVR
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt
from pyHSICLasso import HSICLasso
hsic_lasso = HSICLasso()
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
import warnings
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

#getting the inputs
dataset= input("Enter the name of the dataset: {GSE11223, GSE3365, GSE22619}:")
featureSelection = input("Enter the name of the feature selection method: {mRMR, SVMRFE, HSICLASSO}:")
classifier= input("Which classifier you want to use : {SVM, RF}:")

#******************Reading Dataset************************

start_time = time.time()
print("\nDataset: ", dataset)
print("Loading Dataset") 
# reading the dataset from specified path
data = pd.read_csv("~/..."+dataset+".csv", index_col=0) # index column argument should be removed for GSE11223

#saving the column name to use later
aa=data.columns

#**********************************************************
r,c= data.shape
# fixing the label column and chaning all 0 values to -1 
for i in range (0,r):
    if data.iloc[i,-1]==0:
        data.iloc[i,-1]=-1 
        
#***********************************************
#                Normalization 
#***********************************************
    
#Normalizing GSE3365 since the values are big
#    

if dataset=="GSE3365":
    
    W= data.iloc[:,0:-1]
    P = W.values #returns a numpy arrayc
    min_max_scaler = MinMaxScaler()
    P_scaled = min_max_scaler.fit_transform(P)
    W = pd.DataFrame(P_scaled)
    u=data.iloc[:,-1].to_frame().reset_index()
    data = pd.concat([W , u['class']],axis=1) 
    data.columns=aa
    
#******************Definitions of lists and dataframes********************    

maxiteration=15 
UBOF=20
cols=list(range(0, (UBOF)+1))
idx= list(range(0, (maxiteration)+1))
FeatureAccurcy = pd.DataFrame(index=idx, columns=cols)
recall = pd.DataFrame(index=idx, columns=cols)
specificity = pd.DataFrame(index=idx, columns=cols)
sel_Features= pd.DataFrame(index=list(range(0, (maxiteration))), columns=list(range(0, (UBOF))))
#sel_Features_Names= pd.DataFrame(index=list(range(0, 20)), columns=list(range(0, 5)))
sel_Features_Names= pd.DataFrame(index=list(range(0, 20)), columns=list(range(0, (UBOF))))
auc_df=pd.DataFrame(index=idx, columns=cols)
ap_df=pd.DataFrame(index=idx, columns=cols)

#*****************************************************************************
#             Removing irrelevant features using proposed method (SLS)
#*****************************************************************************    
r,c= data.shape
B=data.iloc[:,-1].to_frame().to_numpy()
A= data.drop(['class'],axis=1)
iA=np.linalg.pinv(A) # Computes the (Moore-Penrose) pseudo-inverse of a matrix
#Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) 
#and including all large singular values. Matrix or stack of matrices to be pseudo-inverted.
Z = np.matmul(iA,B) # Matrix product of two arrays 

if c<200000:
    threshold = max(abs(Z)) - 0.5* max(abs(Z)); # GSE22619 = 0.8, GSE3365 = 0.5, GSE11223= 0.8
else:
    threshold = max(abs(Z)) - .7* max(abs(Z));
    
abs_Z = abs(Z).tolist()
#returning the index of features 
irrF_list =[]
for i in range (len(Z)):
    if abs_Z[i] < threshold:
        irrF_list.append(i)
allF= list(range(0,c))

data_filtered=data.drop(data.columns [irrF_list],axis='columns')   
r1,c1= data_filtered.shape
# Dataset after filtering 
data_filtered = pd.concat([data_filtered.iloc[:,-1] , data_filtered.iloc[:,0:(c1-1)]],axis=1)     
X=data_filtered .drop(['class'],axis=1)
y=data_filtered.loc[:,'class']        

#*****************************************************************************
#                                Cross Validation
#*****************************************************************************
     
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,random_state=36851234)
rskf.get_n_splits(X, y)
j=0

for train, test in rskf.split(X,y):
   
    dataTrain=X.iloc[train,:]
    dataTest=X.iloc[test,:]    
    y_train=y[train]
    y_test=y[test]
        
    print(j+1,"th iteration")
    
           
#********************************************************************
#                      Feature selection
#********************************************************************
        
    print("Running feature selection")
    
#***************************mRMR*************************************   
    
    if featureSelection == "mRMR": 

        df3 = pd.concat([y_train , dataTrain],axis=1)
        selF1 = np.array(pymrmr.mRMR(df3, 'MIQ',UBOF))
        selF=[]
        for s in range(len(selF1)):
            selF.append((selF1[s]))
        
#*********************************SVMRFE*****************************  
                                                                         
    elif featureSelection == "SVMRFE": 
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, UBOF, step=1)
        selector = selector.fit(dataTrain, y_train)
        colsNames=dataTrain.columns
        selR=selector.ranking_
        zipped=pd.DataFrame({'featureNum': colsNames , 'featureRanks': selR})
        selF=[]
        selF_list=[]
        for i in range(0,len(selR)):
            #selF1=[]
            #selF1=np.append(np.array(np.where(selR==1)),selF)
            selF=zipped[zipped.featureRanks==1]
        selF=selF.loc[:,'featureNum']
        selF_list=selF.values.tolist()
        print("Number of selected features:" , len(selF_list))
        
#****************************HSICLASSO********************************      
        
    elif featureSelection=="HSICLASSO":
#       
        df3 = pd.concat([y_train , dataTrain],axis=1)
        #saving the dataset for using with HSICLASSO (first column should be class labels or this method)
        df3.to_csv("~/..." + dataset + "_Lasso.csv", index=False)
        dataLasso =  hsic_lasso.input("~/..." + dataset + "_Lasso.csv")
        hsic_lasso.regression(UBOF)
        R=hsic_lasso.classification(UBOF,B=0,M=1)
        hsic_lasso.dump()
        hsic_lasso.plot_path()
        hsic_lasso.get_index()
        selF1=hsic_lasso.get_features()
        #print("Selected features:" , selF)
        scores=hsic_lasso.get_index_score()
        #print("scores:", scores)
        #Save parameters
        hsic_lasso.save_param()
        selF=[]
        for s in range(len(selF1)):
            selF.append((selF1[s]))

    
#*********************************************************************
#       filtering dataset based on selected features
#*********************************************************************    
    f=len(selF)
    Acc=[]
    recall_list= []
    specificity_list=[]
    auc_list=[]
    ap_list=[]
    for l in range(1,f+1):
    
        selFeatures=selF[0:l+1]       
        X_train=dataTrain.loc[:,selFeatures]
#        y_train=dataTrain.loc[:,'class']
        X_test = dataTest.loc[:,selFeatures]
#        y_test = dataTest.loc[:,'class']
        
#*********************Selecting classifiers****************************************  
        
        if classifier =="SVM" :
            clf = SVC(kernel="linear")
        else: 
            clf=RandomForestClassifier()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train,y_train) 
        y_pred=clf.predict(X_test)
        q1=round(metrics.accuracy_score(y_test, y_pred), 2) * 100
        Acc.append(q1)
        
#*******************AUC Calculation************************************** 
        if classifier =='SVM':
            fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
            
        else:
            fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
            
        auc_=metrics.auc(fpr, tpr)
        auc_list.append(auc_)    
#        print( "CA of SVM: {:.3f}".format(q1), "AUC:{:.2f}".format( metrics.auc(fpr, tpr)))

#************************************************************************
#                  average_precision_score Calculation
#************************************************************************
        
        if classifier =='SVM':
            ap_ = average_precision_score(y_test, clf.decision_function(X_test))
        else:
            ap_ = average_precision_score(y_test, clf.predict_proba(X_test)[:,1])
        ap_list.append(ap_)
#        print("Average precision of svc: {:.3f}".format(ap_svc))
        tn, fp, fn, tp =confusion_matrix(y_test, y_pred, labels=None, sample_weight=None).ravel()
        print("tn,fp,fn,tp: ", "(",tn,",",fp,",",fn,",",tp,")")
    
       #When its "Yes" how often it says "Yes" (Yes=disease)
        l=np.count_nonzero(y_test== 1)
        p=float(tp/l) * 100
        print("Recall: " ,p,"%" )
        recall_list.append(p)
    #        recall[a][l]=p
        #recLyst.append(p)
        
        #when its "No" how often it says "No" ( No=Normal)
        n=np.count_nonzero(y_test== -1)
        n1=float(tn/n) * 100
        specificity_list.append(n1)
    #        specifity[a][l]=n1
        print("Specifity: ", n1,"%\n" )
        print("************************")
        
    for i in range(len(Acc)):
        FeatureAccurcy.iloc[j,i]= Acc[i] 
    selF=pd.DataFrame(selF)   
    for i in range (len(selF)):
        sel_Features_Names.iloc[i,j]= selF.iloc[i,0]   
        
    for i in range(len(Acc)):
        recall.iloc[j,i]=recall_list[i] 
        
    for i in range(len(Acc)):
        specificity.iloc[j,i]= specificity_list[i] 
        
    for i in range(len(auc_list)):
        auc_df.iloc[j,i]= auc_list[i] 
        
    for i in range(len(ap_list)):
        ap_df.iloc[j,i]= ap_list[i] 

    j+=1
    
avgAcc=[]
x=0
#********************************************************************
#              Calculation of Averages for each metric
#********************************************************************

for x in range((UBOF)+1):
    avgAcc.append((FeatureAccurcy.iloc[:,x]).mean())
   
avgRecall=[]
for c in range((UBOF)+1):
    avgRecall.append((recall.iloc[:,c]).mean())
    
avgSpecifity=[]
for b in range((UBOF)+1):
    avgSpecifity.append((specificity.iloc[:,b]).mean())

avg_auc=[]
for b in range((UBOF)+1):
    avg_auc.append((auc_df.iloc[:,b]).mean())

avg_ap=[]
for b in range((UBOF)+1):
    avg_ap.append((ap_df.iloc[:,b]).mean())
    
#**********  Adding averages to corresponding dataframe ************ 
    
for i in range ((UBOF)+1):
    FeatureAccurcy.iloc[-1,i]= avgAcc[i]
    
for i in range ((UBOF)+1):
    recall.iloc[-1,i]= avgRecall[i]  
    
for i in range ((UBOF)+1):
    specificity.iloc[-1,i]= avgSpecifity[i]  
    
for i in range ((UBOF)+1):
    auc_df.iloc[-1,i]= avg_auc[i] 
    
for i in range ((UBOF)+1):
    ap_df.iloc[-1,i]= avg_ap[i]     


maxAvgAccInd= avgAcc.index(max(avgAcc))
accStd=FeatureAccurcy.iloc[0:-1, maxAvgAccInd].std()
recallStd= recall.iloc[0:-1, maxAvgAccInd].std()
specStd= specificity.iloc[0:-1, maxAvgAccInd].std()
aucStd=auc_df.iloc[0:-1, maxAvgAccInd].std()
apStd=ap_df.iloc[0:-1, maxAvgAccInd].std()

#************************** runing time *****************************
    
print("Running Time: %s seconds\n\n" % round((time.time() - start_time),4))

#*************************#printing outputs**************************
print ("maximum mean Ac:", max(avgAcc),"% with standard deviation " , accStd, "for" , maxAvgAccInd+1 , "number of features" )
print(" recall ",  "is", avgRecall[maxAvgAccInd], "% with standard deviation " , recallStd)  
print(" specificity ",  "is", avgSpecifity[maxAvgAccInd], "% with standard deviation ", specStd )
print(" AUC ",  "is", avg_auc[maxAvgAccInd], "% with standard deviation ", aucStd )
print(" Avg_Precision ",  "is", avg_ap[maxAvgAccInd], "% with standard deviation ", apStd)

#saving the name of selected feature to specified path
sel_Features_Names.to_csv("~/.."+ dataset+"_"+featureSelection+"_"+classifier+"_FeatureNames_SLS.csv" )


#*************Average accuracy plot vs feature numbers ************************

x=list(range(0, UBOF+1))
plt.plot(x,avgAcc,label='Average Accuracy')
plt.xticks(x)
plt.xlabel('Number Of Selected Features')
plt.ylabel('Mean classification accuracy')
legend =plt.legend(loc='best', shadow=True, fontsize='small')
plt.savefig("MeanAcc_"+ classifier+"_"+ featureSelection+"_"+ dataset+"_"+ str(maxiteration)+"times.png")
plt.show()


#************Ploting mean Acc, Mean Recall and Mean Specificity in one plot ***

x=list(range(0, UBOF+1))
plt.plot(x,avgAcc,label='Average Accuracy', color='blue')
plt.plot(x,avgRecall, label='Average Recall', linestyle="--", color='lime')
plt.plot(x,avgSpecifity, label='Average Specifity', linestyle=":", color= 'magenta')
plt.xticks(x)
plt.xlabel('Number Of Selected Features')
legend =plt.legend(loc='best', shadow=True, fontsize='small')
plt.savefig("ACC_Recall_Spec_"+ classifier+"_"+ featureSelection+"_"+ dataset+"_"+ str(maxiteration)+"times.png")
plt.show()