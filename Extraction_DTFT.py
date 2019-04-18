# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 08:56:03 2017

@author: baishuhua
"""

# 导入相关库文件

import glob
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ******************** 读取录波数据 ******************** #
def ReadRecord_gd(file): # 读取录波数据，注意解压数据的书写格式
    ReadFlag=open(file)
    segnum=ReadFlag.readline().strip().split(':')[1]
    Time=[]
    for seg in range(int(segnum)):
        SampleAttr=ReadFlag.readline().strip().split(' ')
        [fs,start,terminal]=list(map(int,map(float,SampleAttr)))
        if len(Time)<1:
            Time.extend(1/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
        else:
            Time.extend(Time[-1]+1/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
    ReadFlag.readline() #FaultSeg=int(ReadFlag.readline().strip().split('fault_seg:')[1]) # 故障时刻在哪个分段采样内
    FaultIndex=int(ReadFlag.readline().strip().split('fault_index:')[1]) # 故障时刻索引
    ReadFlag.readline() #FaultNum=int(ReadFlag.readline().strip().split('fault_num:')[1]) # 故障相数目
    FaultPhase=ReadFlag.readline().strip().split('fault_Phase:')[1];FaultPhase.replace(' ','') # 故障类型
    ReadFlag.readline();ReadFlag.readline();ReadFlag.readline()
    
    SignalNames=['ua','ub','uc','u0','ia','ib','ic','i0']  
    Data=[];ValidIndex=[];row=0
    for record in ReadFlag:
        detail=record.strip().split(' ') # 处理空字符串
        detail=[each for each in detail if each!='']
        try:
            Data.append(list(map(float,detail[:8])))
            ValidIndex.append(row)
            row=row+1
        except:
            row=row+1
    ReadFlag.close()
    ValidIndex=np.array(ValidIndex)
    rows=ValidIndex[ValidIndex<len(Time)]    
    Time=np.array(Time);Data=np.array(Data);
    Time=Time[rows];Data=Data[rows]
    return Time,Data,np.array(SignalNames),FaultIndex

# ******************** 显示录波波形 ******************** #        
def ViewWaveForm(x,y,SignalNames):
    fig,axes = plt.subplots(nrows=2, ncols=4, sharex='col', sharey='row')
    for col,ax in enumerate(axes.ravel()):
        ax.plot(x, y[:,col], label=SignalNames[col])
        ax.legend(loc='upper right')
    plt.show()

# ******************** 插值预处理 ******************** #    
def Interp(x,y,fs=1000,kind='slinear',axis=0): # 沿axis轴拟合x，y，因变量y沿axis维度应等于x维度
    function=interpolate.interp1d(x,y,kind,axis)
    x_new=np.arange(min(x),max(x),step=1/fs)
    y_new=function(x_new)
    return x_new,y_new

# ******************** 傅氏提取幅值 ******************** #
def FourierAlgm(circle,orders=[1]):
    N = len(circle) # 一个周波内的采样点数
    Amp, Theta = [], []
    for order in orders:
        coef_real = np.cos(2 * np.pi / N * order * np.arange(1, N + 1))
        coef_imag = np.sin(2 * np.pi / N * order * np.arange(1, N + 1))
        an = 2/N * np.dot(circle, coef_real)
        bn = 2 / N * np.dot(circle, coef_imag)
        if order==0:
            Amp.append(np.sqrt(an ** 2 + bn ** 2)/2)
        else:
            Amp.append(np.sqrt(an ** 2 + bn ** 2))
        Theta.append(np.arctan(-bn/an))
    return np.array(Amp)

# ******************** 抽取单个文件特征 ******************** #
def ExtractingEachFile(file,SelectCol=[4,5,6]): # 此处模拟量顺序应与解压数据一致):
    Time_orig,Data_orig,SignalNames,FaultIndex = ReadRecord_gd(file)
    FaultTime = Time_orig[FaultIndex]
    Partial=(Time_orig>=FaultTime-0.04)&(Time_orig<=FaultTime+0.1)
    Time_orig,Data_orig = Time_orig[Partial],Data_orig[Partial,:]
    Time_new,Data_new = Interp(Time_orig,Data_orig,fs=1000,kind='slinear',axis=0)    
    if (FaultTime>0.02*1.5) & (FaultTime<max(Time_new)-0.02*1.5):
        FeatureEachFile=[];FeatureNameType=[]
        before = (Time_new>=FaultTime-0.02*1.5)&(Time_new<=FaultTime-0.02*0.5)
        after = (Time_new>=FaultTime+0.02*0.5)&(Time_new<=FaultTime+0.02*1.5)
        Data_before = Data_new[before,:][:20,:]
        Data_after = Data_new[after,:][:20,:]
        for col in SelectCol:
            A0 = FourierAlgm(Data_before[:,4], orders=[1]) # 故障前基波
            A1 = FourierAlgm(Data_after[:,col], orders=range(6)) # 故障后各次谐波
            featurevalue=A1/A0
            featurename=[str(i)+'次_'+SignalNames[col] for i in range(len(A1))]
            FeatureEachFile.append(featurevalue)       
            FeatureNameType.append(featurename)
        FeatureEachFile=np.array(FeatureEachFile)
        FeatureNameType=np.array(FeatureNameType)
    return FeatureNameType.ravel(), FeatureEachFile.ravel() # 特征名，特征向量

# ******************** 多文件构建特征集 ******************** #
def ExtractingAllFile(FilePath,SavePath,filename,SelectCol=[4,5,6]):
    BigFeatures=[];BigSamplenames=[];BigLabels=[];BigFeaturenames=None
    SaveFile=os.path.join(SavePath,filename)
    file_object=open(SaveFile,'w',newline='')
    import csv
    writer=csv.writer(file_object)
    
    Lists=glob.glob(FilePath)
    FileNums=len(Lists)
    for FileNo in range(FileNums):
        file=Lists[FileNo]
        filename=os.path.basename(file)
        attr=filename.split('_');FaultCause=attr[0]
            
        try:# 注意SelectCol与SignalNames的对应关系
            featurename,featurevalue=ExtractingEachFile(file,SelectCol=SelectCol)
            
            BigFeatures.append(featurevalue)
            BigSamplenames.append(filename)
            BigLabels.append(FaultCause)
            if BigFeaturenames is None:
                BigFeaturenames = featurename
                head=np.concatenate((np.array(['Filename','Label']),featurename))
                writer.writerow(head)
            line=np.concatenate((np.array([filename,FaultCause]),featurevalue))
            writer.writerow(line)
        except:
            continue
        
    #    if FileNo==0: # 写txt文件
    #        head='Filename\tLabel\t'+'\t'.join(FeatureNameType)+'\n'
    #        file_object.writelines(head)
    #    FeatureEachFile=map(SignificantDigits,FeatureEachFile)
    #    line='\t'.join(map(str,FeatureEachFile))
    #    line=filename+'\t'+FaultCause+'\t'+line+'\n'
    #    file_object.writelines(line)   
        finally:
            print('Completed %.3f%%' % (float(FileNo+1)/FileNums*100))
    BigFeatures=np.array(BigFeatures);BigSamplenames=np.array(BigSamplenames);
    BigLabels=np.array(BigLabels);BigFeaturenames=np.array(BigFeaturenames)
    file_object.close()
    return BigSamplenames,BigFeaturenames,BigFeatures,BigLabels

# ******************** 装载已构建特征集 ******************** #
def DataSet(file):
    import pandas as pd
    Data=pd.read_csv(file,encoding='gbk',na_values=['#NAME?','inf','INF','-inf','-INF'])     
    Featurename=Data.columns.values[2:]                                                
    Data=Data.dropna(how='all',axis=0)
    Data=Data.fillna(axis='index',method='pad')
    FeatureFrame=Data[Featurename]
    Feature=FeatureFrame.values
    
    Filename=Data['Filename'].values
    Label=Data['Label'].values
    return Filename,Featurename,Feature,Label

if __name__=='__main__':    
    if 1:
        FilePath = r'E:\大数据\线路故障诊断\解压后数据\广东中调'
        filename = '对树木放电_2015年11月11日15时47分34秒WYY01.etr-65'
        Time_orig,Data_orig,SignalNames,FaultIndex = ReadRecord_gd(os.path.join(FilePath, filename))
        Partial=(Time_orig>=Time_orig[FaultIndex]-0.04)&(Time_orig<=Time_orig[FaultIndex]+0.1)
        Time_orig,Data_orig = Time_orig[Partial],Data_orig[Partial,:]
        print('******************** 原始波形 ********************\n')
        ViewWaveForm(Time_orig,Data_orig,SignalNames)
        
        print('******************** 插值波形 ********************\n')
        Time_new,Data_new = Interp(Time_orig,Data_orig,fs=1000,kind='slinear',axis=0)
        ViewWaveForm(Time_new,Data_new,SignalNames)

        print('******************** 单个文件提特征 ********************\n')
        file = os.path.join(FilePath,filename)
        featurename,featurevalue = ExtractingEachFile(file,SelectCol=[7])
        
    if 0:
        print('******************** 构建特征集 ********************\n')
        FilePath = r'E:\大数据\线路故障诊断\解压后数据\广东中调\*.etr-*'
        SavePath = r'E:\大数据\线路故障诊断\解压后数据\广东中调'
        SaveFile = '整次谐波特征_零序_gd.csv'
        BigSamplenames,BigFeaturenames,BigFeatures,BigLabels = \
        ExtractingAllFile(FilePath,SavePath,SaveFile,SelectCol=[7])
    
    if 1:
        print('******************** 装载特征集 ********************\n')
        SavePath = r'E:\大数据\线路故障诊断\解压后数据\广东中调'
        SaveFile = '整次谐波特征_零序_gd.csv'
        BigSamplenames,BigFeaturenames,BigFeatures,BigLabels = DataSet(os.path.join(SavePath,SaveFile))

    # 椭圆分布假设的异常检测
    if 0:
        from sklearn import covariance
        contamination = 0.05 # 需设置异常比例
        clf = covariance.EllipticEnvelope(assume_centered=False, support_fraction=None, \
                                          contamination=contamination, random_state=42)
        clf.fit(BigFeatures)
        y_detection=clf.predict(BigFeatures)
        print(BigSamplenames[y_detection==-1])
        
    # 隔离森林异常检测，适于多维数据集
    if 1:
        print('******************** 剔除异常样本 ********************\n')
        from sklearn import ensemble
        contamination = 0.05 # 需设置异常比例
        clf = ensemble.IsolationForest(max_samples='auto', contamination=contamination, \
                                       max_features=1.0, bootstrap=False, random_state=42)
        clf.fit(BigFeatures)
        y_detection=clf.predict(BigFeatures)
        print('异常样本类别:\n',BigSamplenames[y_detection==-1])
        Samplenames,Labels,Features = \
        BigSamplenames[y_detection!=-1],BigLabels[y_detection!=-1],BigFeatures[y_detection!=-1,:]
        
    # OCSVM异常检测,超参数不易设置
    if 0:
        from sklearn import svm
        clf = svm.OneClassSVM(kernel='rbf', nu=0.5, max_iter=-1, random_state=42)
        clf.fit(BigFeatures)
        y_detection=clf.predict(BigFeatures)
        print(BigSamplenames[y_detection==-1])
    
    if 1:   
        print('******************** 可视化特征集 ********************\n')             
        import seaborn as sn
        import pandas as pd
        Features = Features.astype(np.float64)
        frame = pd.DataFrame(np.column_stack((Features,Labels)), \
                             columns=np.append(BigFeaturenames,'category'))
        # 三类特征两两散点图及直方图
    #    vars_view = [name for name in BigFeaturenames if name.startswith('3次')]     
    #    sn.pairplot(frame, hue='category', kind='scatter', diag_kind='kde',\
    #                plot_kws=dict(alpha=0.7), palette="husl") # vars=vars_view,        
        for i in range(Features.shape[1]):
            plt.figure()
            sn.violinplot(x=i*np.ones(len(Features)), y=Features[:,i], data=frame, \
                          hue='category', scale='area', inner='quartile')
            plt.title(BigFeaturenames[i])
        

#        