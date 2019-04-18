# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:17:36 2017
训练分类器
@author: baishuhua
"""
import warnings
warnings.filterwarnings('ignore')
import time

import os

import Extraction_DTFT as Ext
import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline

random_state=np.random.seed()

import pickle

import sys;sys.path.append('E:\\大数据') 
import sampler
from collections import Counter

# ******************** 装载已构建特征集 ******************** #
def DataSet(file):
    Data=pd.read_csv(file,encoding='gbk',na_values=['#NAME?','inf','INF','-inf','-INF'])     
    Featurename=Data.columns.values[2:]                                                
    Data=Data.dropna(how='all',axis=0,subset=Featurename)
    Data=Data.fillna(axis='index',method='pad')
    Feature=Data[Featurename].values
    Filename=Data['Filename'].values
    Label=Data['Label'].values
    return Filename,Featurename,Feature,Label

# ******************** 剔除异常样本 ******************** #
def RemoveAbnormal(BigFeatures,contamination = 0.05):
    print('******************** 剔除异常样本 ********************\n')
    from sklearn import ensemble
    clf = ensemble.IsolationForest(max_samples='auto', contamination=contamination, \
                                   max_features=1.0, bootstrap=False, random_state=42)
    clf.fit(BigFeatures)
    y_detection=clf.predict(BigFeatures)
    mask = (y_detection==-1)
    return mask # 异常样本编号        
        
def Encoding(Label):
    Encoder=preprocessing.LabelEncoder().fit(Label)
    return Encoder

# ******************** 标准化处理特征集 ******************** #
def Scaler():
    scaler=preprocessing.MaxAbsScaler(copy=True)
    return scaler

# ******************** 模型评估 ******************** #
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,f1_score,fbeta_score,jaccard_similarity_score,precision_score,recall_score # 该类指标越大性能越好
from sklearn.metrics import hamming_loss,zero_one_loss  # 该类指标绝对值越小性能越好
def Evaluate(score_func=accuracy_score):
    if score_func in [accuracy_score,f1_score,fbeta_score,jaccard_similarity_score,precision_score,recall_score]:
        greater_is_better=True
    elif score_func in [hamming_loss,zero_one_loss]:
        greater_is_better=False
    score=make_scorer(score_func=score_func,greater_is_better=greater_is_better,needs_proba=False,needs_threshold=False,average='micro')#
    return score

# 混淆矩阵可视化预测结果
import itertools
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    plt.figure(figsize=(12,6))
    plt.imshow(cm,interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    if normalize:
        cm=cm.astype('float')/(cm.sum(axis=1)[:,np.newaxis]+1e-10)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    thresh=cm.max()/ 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # itertools.product笛卡尔积   代表两层（嵌套循环）
        plt.text(j,i,round(cm[i,j],2),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签') # True label
    plt.xlabel('预测标签') # Predicted label

# 可视化二分类器的ROC曲线，并计算曲线积分面积
def Plot_ROC_binary(ys_true,ys_scores,view):
    y_true=ys_true;y_score=ys_scores
    fpr,tpr,thresholds=roc_curve(y_true,y_score,pos_label=None,sample_weight=None,drop_intermediate=True)
    area=auc(fpr,tpr,reorder=False)
    if view:
        plt.figure()
        plt.plot(fpr,tpr,label='ROC and AUC is %.2f' %area)
        plt.legend(loc='lower right')
        plt.xlim([0.0,1.0]);plt.ylim([0.0,1.1])
        plt.xlabel('假正类');plt.ylabel('真正类')
        plt.title('ROC曲线')
        plt.show()
    return area
 
# 可视化多分类器的ROC曲线，并加权计算曲线积分面积
def Plot_ROC_multi(ys_orig,ys_true,ys_scores,classnames,view):
    fpr=dict();tpr=dict();roc_auc=dict()
    for i,classname in enumerate(classnames):
        y_true=ys_true[:,i];y_score=ys_scores[:,i]
        fpr[i],tpr[i],thresholds=roc_curve(y_true,y_score,pos_label=None,sample_weight=None,drop_intermediate=True)
        area=auc(fpr[i],tpr[i],reorder=False)
        roc_auc[i]=area    
    
    # Compute weighted-average ROC curve and ROC area,对于weighted模式，各类的加权系数为各类在总类中的占比
    all_fpr=np.unique(np.concatenate([fpr[i] for i in range(len(fpr))]))
    # Then interpolate all ROC curves at this points
    mean_tpr=np.zeros_like(all_fpr)
    from scipy import interp
    for i in range(len(classnames)):
        mean_tpr += interp(all_fpr,fpr[i],tpr[i])*(sum(ys_orig==i)/len(ys_orig))
    # Finally compute AUC
    fpr['weighted']=all_fpr;tpr['weighted']=mean_tpr
    roc_auc['weighted']=auc(fpr['weighted'],tpr['weighted'])
    
    if view:
        plt.figure(figsize=(8,6))
        for i,classname in enumerate(classnames):
            plt.plot(fpr[i],tpr[i],label=classname+' ROC and AUC is %.2f' %(roc_auc[i]))
        plt.plot(fpr['weighted'],tpr['weighted'],label='综合 ROC and AUC is %.2f' %(roc_auc['weighted']))
        plt.legend(loc='lower right')
        plt.xlim([0.0,1.0]);plt.ylim([0.0,1.1])
        plt.show()
    
    return roc_auc['weighted']

if __name__=='__main__':
    IsExt = 0
    if IsExt:
        print('******************** Start extracting features ********************\n')
        FilePath = r'E:\大数据\线路故障诊断\解压后数据\广东中调\*.etr-*'
        SavePath = r'E:\大数据\线路故障诊断'
        SaveFile = '故障整次谐波_三相_gd.csv'
        BigSamplenames,BigFeaturenames,BigFeatures,BigLabels = \
        Ext.ExtractingAllFile(FilePath,SavePath,SaveFile,SelectCol=[4,5,6])
    else:
        print('******************** Load extracted features ********************\n')
        SavePath = r'E:\大数据\线路故障诊断'
        SaveFile = '故障整次谐波_零序_gd.csv'
        BigSamplenames,BigFeaturenames,BigFeatures,BigLabels = DataSet(os.path.join(SavePath,SaveFile))
    
    # 装载matlab提取的小波能量均值、标准差和能量熵特征
    if 1:
        from scipy.io import loadmat 
        BigFeatures = loadmat(r'E:\大数据\线路故障诊断\XY.mat')['Features'] # 3指标的对数
        BigLabels = loadmat(r'E:\大数据\线路故障诊断\XY.mat')['Labels'][0] # 注意数据格式
        BigLabels = np.array([element[0] for element in BigLabels])
        BigSamplenames = loadmat(r'E:\大数据\线路故障诊断\XY.mat')['Files'][0] # 注意数据格式
        BigSamplenames = np.array([element[0] for element in BigSamplenames])
    
    IsRemove = 1
    if IsRemove:
        mask = RemoveAbnormal(BigFeatures,contamination = 0.05)        
        print('异常样本类别:\n',BigSamplenames[mask])
        Samplenames,Labels,Features = BigSamplenames[~mask],BigLabels[~mask],BigFeatures[~mask,:]
    else:
        Samplenames,Labels,Features = BigSamplenames,BigLabels,BigFeatures
        
    # 筛选
    select = (Labels=='山火') | (Labels=='外力破坏') 
    Labels = Labels[select]
    Features = Features[select]
    
    # 编码
    Encoder = Encoding(Labels)
    Y = Encoder.transform(Labels)
    X = Features.astype(np.float64)
    
    # 拆分训练集和测试集
    seed = 42 #np.random.seed()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print('TrainSet details:\n',Counter(Encoder.inverse_transform(Y_train)))
    print('TestSet details:\n',Counter(Encoder.inverse_transform(Y_test)))
    
    # 均衡化样本集
    IsBalanced = 1
    if IsBalanced:
        X_train,Y_train = sampler.UnderSample(X_train, Y_train, method='Random', random_state=seed)
    
    
    score_func = f1_score
    # 随机分类器
    if 1:
        print(' '.join(['*'*25,'simple rules','*'*25,'\n']))        
        clf_dummy=DummyClassifier(strategy='uniform',random_state=seed,constant=None)
        pipe_dummy=Pipeline([('scaler',Scaler()),('clf',clf_dummy)]) # ('reduct',MyLDA()),
        start=time.time()
        pipe_dummy=pipe_dummy.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        Y_pred=pipe_dummy.predict(X_test)
        #print('Predict Results: ',Encoder.inverse_transform(Y_pred))
        #print('True Results: ',Encoder.inverse_transform(Y_test))
        judge=cross_val_score(pipe_dummy,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean()))        
                
    # 默认超参数SVM分类器
    if 0:
        print(' '.join(['*'*25,'svm','*'*25,'\n']))
        clf_svm=SVC(C=10,kernel='rbf',gamma=0.1,probability=True,\
                decision_function_shape='ovr',random_state=seed,class_weight='balanced') #     
        pipe_svm=Pipeline([('scaler',Scaler()),('clf',clf_svm)]) 
        start=time.time()
        pipe_svm=pipe_svm.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        Y_pred=pipe_svm.predict(X_test)
        #print('Predict Results: ',Encoder.inverse_transform(Y_pred))
        #print('True Results: ',Encoder.inverse_transform(Y_test))
        judge=cross_val_score(pipe_svm,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean()))
        
    # 网格搜索超参数SVM
    if 1:
        print(' '.join(['*'*25,'svm + GridSearch','*'*25,'\n']))
        clf_svm0=SVC(C=10,kernel='rbf',gamma=0.1,probability=True,\
                decision_function_shape='ovr',random_state=seed,class_weight='balanced')        
        pipe_svm0=Pipeline([('scaler',Scaler()),('clf',clf_svm0)])
        param_grid = [{'clf__C':np.logspace(-5,5,num=11),'clf__gamma': np.logspace(-5,5,num=11)}]
        grid_search=GridSearchCV(estimator=pipe_svm0,param_grid=param_grid,scoring=Evaluate(score_func),\
                                     fit_params=None,n_jobs=1,refit=True,cv=3,return_train_score=True)
        start=time.time()
        grid_search=grid_search.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        pipe_svm0=grid_search.best_estimator_
        #print("Best parameters: {}".format(grid_search.best_params_))
        #print("Best cross-validation score: {}".format(grid_search.best_score_))
        judge=cross_val_score(pipe_svm0,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) # 可替换成与交叉验证相同的评价指标，待更新
        
    # Bagging method： Random Forest        
    if 1:
        print(' '.join(['*'*25,'RandomForestClassifier','*'*25,'\n']))  
        from sklearn.ensemble import RandomForestClassifier
        clf_rdft = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2, \
                                           bootstrap=True, oob_score=False, random_state=seed, \
                                           class_weight='balanced')
        rdft = Pipeline([('scaler',Scaler()),('clf',clf_rdft)])
        start=time.time()
        rdft = rdft.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        judge=cross_val_score(rdft,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) # 可替换成与交叉验证相同的评价指标，待更新
        
        if 0:
            try:
                # 特征重要性
                feature_importance = rdft.steps[1][1].feature_importances_
                # make importances relative to max importance
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(sorted_idx.shape[0]) + .5
                plt.figure()
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                plt.yticks(pos, BigFeaturenames[sorted_idx])
                plt.xlabel('Relative Importance')
                plt.title('Variable Importance based on bagging method')
                plt.show()
            except:
                print('不展示特征重要性')
    
    if 0:
        print(' '.join(['*'*25,'RandomForestClassifier','*'*25,'\n']))  
        from sklearn.ensemble.bagging import BaggingClassifier
        clf_svm0=SVC(C=10,kernel='rbf',gamma=0.1,probability=True,\
                decision_function_shape='ovr',random_state=seed,class_weight='balanced')        
        pipe_svm0=Pipeline([('scaler',Scaler()),('clf',clf_svm0)])
        clf_bg = BaggingClassifier(base_estimator=pipe_svm0, n_estimators=10, max_samples=1.0, \
                                   max_features=1.0, random_state=seed)
        start=time.time()
        clf_bg = clf_bg.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        judge=cross_val_score(clf_bg,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) 
        
    # Boosting method：GradBoost
    if 1:
        print(' '.join(['*'*25,'GradientBoostingClassifier','*'*25,'\n'])) 
        from sklearn.ensemble import GradientBoostingClassifier
        clf_gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, \
                                            subsample=1.0, criterion='friedman_mse', 
                                            min_samples_split=2, max_depth=3, random_state=seed)
        pipe_gb=Pipeline([('scaler',Scaler()),('clf',clf_gb)]) 
        start=time.time()
        pipe_svm = pipe_gb.fit(X_train,Y_train)
        print('Total running time is {}s'.format(time.time()-start))
        judge=cross_val_score(pipe_gb,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) 
        
        # 特征重要性
        if 0:
            try:
                feature_importance = pipe_gb.steps[1][1].feature_importances_
                # make importances relative to max importance
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                pos = np.arange(sorted_idx.shape[0]) + .5
                plt.figure()
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                plt.yticks(pos, BigFeaturenames[sorted_idx])
                plt.xlabel('Relative Importance')
                plt.title('Variable Importance based on boosting method')
                plt.show()
            except:
                print('不展示特征重要性')    
                
    # 人工神经网络
    if 1:
        print(' '.join(['*'*25,'MLPClassifier','*'*25,'\n'])) 
        from sklearn.neural_network import MLPClassifier    
        #solver可取'lbfgs' 'sgd' 'adam'
        clf_mlp = MLPClassifier(hidden_layer_sizes=(100,100,), activation='relu', solver='adam', alpha=0.0001, \
                            batch_size='auto', learning_rate='constant', learning_rate_init=0.001, \
                            power_t=0.5, max_iter=200, shuffle=True, random_state=seed, \
                            validation_fraction=0.1)
        pipe_mlp=Pipeline([('scaler',Scaler()),('clf',clf_mlp)])     
        start=time.time()
        pipe_mlp = pipe_mlp.fit(X_train, Y_train)  
        print('Total running time is {}s'.format(time.time()-start))
        judge=cross_val_score(pipe_mlp,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
        #print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean()))     

 

#    
#    
#    # 混淆矩阵评估
#    if 1:
#        from sklearn.metrics import confusion_matrix
#        y_true = Y_test
#        y_pred = pipe.predict(X_test)
#        cnf_matrix=confusion_matrix(y_true,y_pred,labels=np.unique(Y))
#        classes = Encoder.inverse_transform(np.unique(Y))
#        plot_confusion_matrix(cnf_matrix,classes,normalize=False,title='混淆矩阵') 
#    # ROC评估
#    if 0:
#        from sklearn.metrics import roc_curve,auc,roc_auc_score
#        ys_true = Y_new
#        ys_scores = pipe.decision_function(X_new)
#        # 二分类问题
#        area = Plot_ROC_binary(ys_true,ys_scores,view=1)
#    if 0:
#        from sklearn.preprocessing import LabelBinarizer
#        from sklearn.metrics import roc_curve,auc
#        
#        lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False).fit(Y)        
#        classes = lb.classes_
#        classnames = Encoder.inverse_transform(classes)
#        
#        YY=lb.transform(Y)
#        XX = X
#        XX_train,XX_test,YY_train,YY_test=train_test_split(XX,YY,test_size=0.2,train_size=0.8,random_state=random_state)
#            
#        from sklearn.multiclass import OneVsRestClassifier        
#        estimator=OneVsRestClassifier(pipe,n_jobs=1)
#        estimator=estimator.fit(XX_train,YY_train)
#
#        ys_scores = estimator.decision_function(XX_test)
#        Y_test = lb.inverse_transform(YY_test)
#        # Y_test代表多分类标签列表，YY_test代表真实的二进制编码后的标签数组，ys_scores为决策输出
#        # classnames为二进制编码每列对应的类别标记
#        area = Plot_ROC_multi(Y_test,YY_test,ys_scores,classnames,1)
#        
#     
#    if 1: # 打印列表，按列分别为真实类别、预测类别、所属于各类的概率
#        aaa1=y_true.reshape(-1,1)
#        aaa2=y_pred.reshape(-1,1)
#        aaa3=pipe.predict_proba(X_new)
#        aaa=np.concatenate((aaa1,aaa2,aaa3),axis=1)
#        print(aaa)
#        
##        # ######################### 粒子群优化 #########################
##        if 0:
##            print(' '.join(['*'*25,model,'and','PSO','*'*25]))
##            start=time.time()
##            lb=[1e-3,1e-3];ub=[1e2,1e2]
##            xopt,fopt=pso(object0,lb,ub,omega=0.2,phip=0.5,phig=0.5,maxiter=100,minstep=1e-2,minfunc=1e-8)
##            ##print("Best parameters: {}".format(xopt))
##            ##print("Best cross-validation score: {}".format(fopt*-1))
##            clf=Classifier(model='svm')
##            clf.set_params(C=xopt[0],gamma=xopt[1])
##            pipe=Pipeline([('scaler',Scaler()),('reduct',MyPCA()),('clf',clf)])     
##            pipe=pipe.fit(X_train,Y_train)
##            print('Score is {}'.format(pipe.score(X_test,Y_test)))
##            judge=cross_val_score(pipe,X,Y,groups=None,scoring=Evaluate(score_func),cv=5)
##            #print('Cross-validation score is {}'.format(judge))
##            print('Mean cross-validation score is {}'.format(judge.mean()))
##            print('Total running time is {}s'.format(time.time()-start))
##    
##        # ######################### 模型保存及再利用 #########################
##        pickle.dump(pipe,open('model.sav','wb'))
##        model=pickle.load(open('model.sav','rb'))
###        scores.append(model.score(X_test,Y_test))
#
#    ########## 利用新样本测试预测效果
#    if 0:
#        newfile=r'E:\大数据\线路故障诊断\Feature_zj.csv'
#        newFilename,newFeaturename,newFeature,newLabel=DataSet(newfile)
#        
#        partial=(newLabel!='覆冰')
#        newX=np.array(newFeature[partial],dtype=np.float32)
#        newY=Encoder.transform(newLabel[partial])      
#        print('Score is {}'.format(pipe.score(newX,newY)))
#    
    
        
# SVM 相关参数描述
# 核函数为‘linear’参数包括：
# 核函数为poly参数包括：degree gamma  coef0  
# 核函数为rbf参数包括：gamma  
# 核函数为sigmoid参数包括：gamma  coef0

# 通用设置参数包括：C  kernel  probability  shrinking  tol  class_weight  cache_size  verbose  max_iter  
# decision_function_shape  ramdom_state，其中，往往以下参数采用默认即可：probability  shrinking  tol  
# cache_size  verbose  max_iter  主要需设置的参数: C kernel class_weight  decision_function_shape random_state
