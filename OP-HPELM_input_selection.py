import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import sys
from scipy.stats import norm
import hpelm
import math

#反归一化
def Anti_Normalization(origin, normalization):
  
    #assert len(origin) == len(normalization)
    Result = []  
    
    for i in range(len(normalization)):
        Result.append(normalization[i]*(origin.max()-origin.min())+origin.min())
    return Result

#SFS检验原则模块（AIC准则）  （from wikipedia）Burnham, K. P.; Anderson, D. R. (2002), Model Selection and Multimodel Inference: A practical information-theoretic approach (2nd ed.), Springer-Verlag, ISBN 0-387-95364-7.
def calcAIC(observed_ori, predicted,n,p): #n是样本量， p是参数量，RSS是反归一化后得出的
	#反归一化
	predicted_anti=Anti_Normalization(observed_ori, predicted)

	RSS=sum([i**2 for i in ((predicted_anti-observed_ori))])
	AIC=n*(math.log(RSS/n))+2*(p+1)**2#log(n)

	return AIC  

#基于hpelm的极限学习机(可防训练不收敛导致的错误出现)
def extreme_learning_machine(input, observed, dimensions_of_input, Norm=10**-2, num=30):#input是ndarray；默认情况下无L2正则化
	
	assert len(input) == len(observed)
	
	def Train_func():
		try:
			model.train(input, observed, 'r', 'OP', 'LOO')
		except:
			return 0, 0
		else:
			return 1, model

	#创建模型
	forecast_result0=[]
	for i in range(num):
		Train_result=1
		#如果某次训练不收敛，则会重复训练直至收敛
		while (Train_result):
		
			model=hpelm.ELM(dimensions_of_input, 1, norm=Norm)#输入特征个数:dimensions_of_input
			model.add_neurons(100, 'sigm')#lin, sigm, tanh, rbf_l1, rbf_l2, rbf_linf
			#model.add_neurons(50, 'rbf_l1')#lin, sigm, tanh, rbf_l1, rbf_l2, rbf_linf
			#model.add_neurons(50, 'rbf_l2')#lin, sigm, tanh, rbf_l1, rbf_l2, rbf_linf
			#model.add_neurons(50, 'rbf_linf')#lin, sigm, tanh, rbf_l1, rbf_l2, rbf_linf		
			Train_result=Train_func()
			
			if Train_result[0]==1:
				forecast_result0.append(Train_result[1].predict(input).flatten().tolist())
				Train_result=0
			else:
				Train_result=1
			
		
	forecast_result0=np.array(forecast_result0)
		
	forecast_result=forecast_result0.sum(axis=0)/num
	
	return model, forecast_result
	
def SFS(input,observed,benchmark,observed_ori):
  
  import time
  
  time_start=time.clock()
  
  number=[]
  selected=[]
  big_big_transfer=[]

  nan_array=[]
  for i in range(len(input[0])):
     nan_array.append(np.nan)
  nan_array=np.array(nan_array) 

  judge=1; n_round=0

  while judge==1 or n_round==0:

    big_transfer=[]
 
    if n_round==0:

        transfer_factor=[]
        for i in range(len(input)):
            transfer_factor.append(input[i])#=big_big_transfer[0][0][i]  (array)
        big_transfer.append(transfer_factor)#0.预报因子

        transfer_forecast=[]
        #pdb.set_trace()
        for i in range(len(input)):
            transfer_forecast.append(extreme_learning_machine(big_transfer[0][i].reshape(len(input[0]),1),observed, 1)[1].flatten())#=big_big_transfer[0][1][i]
        #pdb.set_trace()
        big_transfer.append(transfer_forecast)#1.预报结果

    else:

        t=n_round-1

        number.append(selected[t])#上一轮挑出的因子

        transfer_factor=[]
        for i in range(len(input)):
          if n_round==1:
                transfer_factor.append(np.hstack((input[i].reshape(len(input[0]),1),big_big_transfer[t][0][number[t]].reshape(len(input[0]),1))).tolist())
          else:
                transfer_factor.append(np.hstack((input[i].reshape(len(input[0]),1),big_big_transfer[t][0][number[t]])).tolist())

        big_transfer.append(np.array(transfer_factor))
        
        for i in range(n_round):
            big_transfer[0][number[i]]=np.nan#将上一轮挑选的因子剔除掉
   
        transfer_forecast=[]
        for i in range(len(input)):
            if np.isnan(big_transfer[0][i][0][0]):
                 transfer_forecast.append(nan_array)
            else: 
                 transfer_forecast.append(extreme_learning_machine(big_transfer[0][i], observed, len(big_transfer[0][i][0]))[1].flatten())
        #pdb.set_trace()
        big_transfer.append(transfer_forecast)

   
   
    transfer_forecast_err=[]  
    for i in range(len(input)):
        transfer_forecast_err.append(big_transfer[1][i]-observed.tolist())
    big_transfer.append(transfer_forecast_err)#2.预报误差

	
	
    if n_round==0:
	
        transfer_cri_score=[]
        for i in range(len(input)):
            transfer_cri_score.append(calcAIC(observed_ori, big_transfer[1][i],len(observed_ori),n_round+1))
        big_transfer.append(transfer_cri_score)#3.标准得分
	
    else:
	
        transfer_cri_score=[]
        for i in range(len(input)):
            transfer_cri_score.append(calcAIC(observed_ori, big_transfer[1][i],len(observed_ori),n_round+1)) 
        big_transfer.append(transfer_cri_score)
	


   #最优因子挑选(based on AE)
    transfer_AE_set=[]; transfer_AE_range=[]
	
    if n_round==0:
        for i in range(len(input)):
            transfer=big_transfer[3][i]
            if transfer<calcAIC(observed_ori, benchmark,len(observed_ori),n_round):
                transfer_AE_set.append(transfer); transfer_AE_range.append(i)
        big_transfer.append(transfer_AE_set)#4.小于前一标准的AIC们
        big_transfer.append(transfer_AE_range)#5.小于前一标准的AIC的编号们
		
    else:
        for i in range(len(input)):
            transfer=big_transfer[3][i]
            if transfer<big_big_transfer[t][3][number[t]]:
                transfer_AE_set.append(transfer); transfer_AE_range.append(i)
        big_transfer.append(transfer_AE_set)
        big_transfer.append(transfer_AE_range)

    try:   
        big_transfer[5][big_transfer[4].index(min(big_transfer[4]))]
    except ValueError:
        judge=0
    else:
        judge=1; selected.append(big_transfer[5][big_transfer[4].index(min(big_transfer[4]))])

    
    big_big_transfer.append(big_transfer)
    n_round+=1

	   #big_big_transfer[i]为i轮变量; big_big_transfer[i][i]为i轮第i个变量
  time_end=time.clock()
  
  return n_round, judge, selected, big_big_transfer, time_end-time_start
  
  
def SFS2(input0,input,observed,benchmark,observed_ori):#第一分组以后的挑选
 
#1.第一回合挑选的因子是组合因子；
#2.之后，恢复到正常因子集；
#3.每组至少要挑选出一个因子

#a.第一回合有没有选出因子？
#		NO：至少要挑选出一个因子；Yes：继续第一回合之后的正常流程
 
  import time
  
  time_start=time.clock()
  
  number=[]
  selected=[]
  big_big_transfer=[]

  nan_array=[]
  for i in range(len(input[0])):
     nan_array.append(np.nan)
  nan_array=np.array(nan_array) 

  judge=1; n_round=0

  while judge==1 or n_round==0:#缺少一个上限限制

    big_transfer=[]
 
    if n_round==0:

        transfer_factor=[]
        for i in range(len(input0)):
            transfer_factor.append(input0[i])#=big_big_transfer[0][0][i]  (array)
        big_transfer.append(transfer_factor)

        transfer_forecast=[]
        #pdb.set_trace()
        for i in range(len(input0)):
            transfer_forecast.append(extreme_learning_machine(big_transfer[0][i] ,observed, len(big_transfer[0][i][0]))[1].flatten())#=big_big_transfer[0][1][i]; big_transfer[0][i]已默认已经完成转换
            IV_num=len(big_transfer[0][i][0])
		#pdb.set_trace()
        big_transfer.append(transfer_forecast)

    else:

        t=n_round-1

        number.append(selected[t])#上一轮挑出的因子

        transfer_factor=[]
        for i in range(len(input)):
            transfer_factor.append(np.hstack((input[i].reshape(len(input[0]),1),big_big_transfer[t][0][number[t]])).tolist())#NO two reshapes!

        big_transfer.append(np.array(transfer_factor))
        
        for i in range(n_round):
            big_transfer[0][number[i]]=np.nan
   
        transfer_forecast=[]
        for i in range(len(input)):
            if np.isnan(big_transfer[0][i][0][0]):
                 transfer_forecast.append(nan_array)
            else: 
                 transfer_forecast.append(extreme_learning_machine(big_transfer[0][i], observed, len(big_transfer[0][i][0]))[1].flatten())
                 IV_num=len(big_transfer[0][i][0])

        #pdb.set_trace()
        big_transfer.append(transfer_forecast)
   
    transfer_forecast_err=[]  
    for i in range(len(input)):
        transfer_forecast_err.append(big_transfer[1][i]-observed.tolist())
    big_transfer.append(transfer_forecast_err)

	
	
    if n_round==0:
	
        transfer_cri_score=[]
        for i in range(len(input)):
            transfer_cri_score.append(calcAIC(observed_ori, big_transfer[1][i],len(observed_ori),IV_num))
        big_transfer.append(transfer_cri_score)#3.标准得分
	
    else:
	
        transfer_cri_score=[]
        for i in range(len(input)):
            transfer_cri_score.append(calcAIC(observed_ori, big_transfer[1][i],len(observed_ori),IV_num))
        big_transfer.append(transfer_cri_score)
	


   #最优因子挑选(based on AE)
    transfer_AE_allset=[]
    transfer_AE_set=[]; transfer_AE_range=[]
	
    if n_round==0:
        for i in range(len(input)):
            transfer_AE_allset.append(big_transfer[3][i])
            transfer=big_transfer[3][i]
            if transfer<calcAIC(observed_ori, benchmark,len(observed_ori),IV_num-1):    
                transfer_AE_set.append(transfer); transfer_AE_range.append(i)
        big_transfer.append(transfer_AE_set)#4.大于前一标准得分的得分们
        big_transfer.append(transfer_AE_range)#5.大于前一标准得分的得分编号们
		
    else:
        for i in range(len(input)):
            transfer_AE_allset.append(big_transfer[3][i])
            transfer=big_transfer[3][i]
            if transfer<big_big_transfer[t][3][number[t]]:
                transfer_AE_set.append(transfer); transfer_AE_range.append(i)
        big_transfer.append(transfer_AE_set)
        big_transfer.append(transfer_AE_range)


    try:   
        big_transfer[5][big_transfer[4].index(min(big_transfer[4]))]
    except ValueError:
        judge=0#说明第0轮没有选到！
        if n_round==0:
           big_transfer[4].append(min(transfer_AE_allset))
           big_transfer[5].append(transfer_AE_allset.index(min(transfer_AE_allset)))
           selected.append(transfer_AE_allset.index(min(transfer_AE_allset)))
    else:
        judge=1; selected.append(big_transfer[5][big_transfer[4].index(min(big_transfer[4]))])
        print(IV_num)
   
	
    big_big_transfer.append(big_transfer)
    n_round+=1

	   #big_big_transfer[i]为i轮变量; big_big_transfer[i][i]为i轮第i个变量
  time_end=time.clock()
  
  return n_round, judge, selected, big_big_transfer, time_end-time_start
 
#Benchmark1=IRWA(Data[2],Data[3],benchmark)

#input是一个分组的list（顺序自定），首先挑选第一组得到n个因子，或者是一个因子（当第一次的benchmark结果好于第一组任意一个结果时），将得到的因子与第二组的每个因子
#        进行组合（hstack），再将第一组产生的最好结果作为benchmark，继续得到第二组产生的因子...要在结果标记出选出的因子是否能够提升拟合效果。


def Group_selection(input,observed,benchmark,observed_ori):#input是一个分组的list（顺序自定）

	import time
	
	time_start=time.clock()
	#将所有因子进行变换
	A=[]
	for i in range(len(input)):
		B=[]
		for j in range(len(input[i])):
			length=len(input[i][j])
			B.append(input[i][j].reshape(length,1))
		A.append(B)
	
	factor_selection=[]
	Selection=[]
	Benchmark=benchmark
	D=[]
	i=0
	while i<len(input):
		
		if i==0:
			Selection.append(SFS(input[i],observed,Benchmark,observed_ori))#第一轮
			
		else:#其它轮
			optimal=Selection[i-1][2][-1]
			if len(Selection[i-1][3])>1:
				Benchmark=Selection[i-1][3][-2][1][optimal]#是上一轮被对比的预报值
			else:
				Benchmark=Selection[i-1][3][0][1][optimal]#是上一轮被对比的预报值
			Selection.append(SFS2(input0,input[i],observed,Benchmark,observed_ori))
		
		if len(Selection[i][2])>1:#当选出的因子大于1个
			C=np.hstack((A[i][Selection[i][2][0]],A[i][Selection[i][2][1]]))
			for j in range(len(Selection[i][2])-2):
				C=np.hstack((C,A[i][Selection[i][2][j+2]]))#本轮因子集组合

			if i>0:
				C=np.hstack((C,D[i-1]))
	
				
		else:#当只选出1个因子
			C=A[i][Selection[i][2][0]]
			if i>0:
				C=np.hstack((C,D[i-1]))
		
		D.append(C)
			
		if i<(len(input)-1):
			input0=[]
			for k in range(len(A[i+1])):
				input0.append(np.hstack((C,A[i+1][k])))#当前最好的因子集与下一轮因子结合，得到初选潜在因子集
		
		factor_selection.append(Selection[i][2])
		i=i+1
		
	time_end=time.clock()

	return factor_selection, Selection, time_end-time_start, D, C
	

benchmark=np.linspace(-1,1,1107)
Selection_result=Group_selection(Data[0],Data[1],benchmark,Data[8])













