import pandas as pd
import numpy as np
import hpelm

NUM=30 #模型运行重复次数
#基于hpelm的极限学习机(可防训练不收敛导致的错误出现)
def extreme_learning_machine(input, observed, dimensions_of_input, Norm=10**-2, num=NUM):#input是ndarray；默认情况下无L2正则化;  num是重复次数
	
	assert len(input) == len(observed)
	
	model_contain=[]
	
	def Train_func():
		try:
			model.train(input, observed, 'r' , 'OP', 'LOO')
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
			Train_result=Train_func()
			
			if Train_result[0]==1:
				forecast_result0.append(Train_result[1].predict(input).flatten().tolist())
				
				model_contain.append(Train_result[1])
				Train_result=0
						
			else:
				Train_result=1
			
		
	forecast_result0=np.array(forecast_result0)
		
	forecast_result=forecast_result0.sum(axis=0)/num
	
	return model_contain, forecast_result

	

def HPELM_1lead():
	
	A=[]
	length=len(Data[2][0][0])
	for i in range(len(Selection_result[0])):
		for j in (Selection_result[0][i]):
			A.append(Data[2][i][j].reshape(length,1))
	
	B=[]
	for i in range(len(A)):
		if i==0:
			B.append(A[0])
		else:
			C=np.hstack((A[0],A[1]))
			for j in range(i+1-2):
				C=np.hstack((C,A[j+2]))
			B.append(C)
			
	D=[]
	length=len(Data[4][0][0])
	for i in range(len(Selection_result[0])):
		for j in (Selection_result[0][i]):
			D.append(Data[4][i][j].reshape(length,1))
	
	E=[]
	for i in range(len(D)):
		if i==0:
			E.append(D[0])
		else:
			F=np.hstack((D[0],D[1]))
			for j in range(i+1-2):
				F=np.hstack((F,D[j+2]))
			E.append(F)

	elm_result=[]
	moni_tr_result=[]
	evaluation1_all=[]
	moni_te_result=[]
	evaluation2_all=[]
	
	i=0
	while i<len(B):		
		elm_result.append(extreme_learning_machine(B[i], Data[3], len(B[i][0])))
		
		moni_tr_result.append(Anti_Normalization(Data[8], elm_result[i][1]))
	
		evaluation1=[]
		evaluation1.append(r(Data[6], moni_tr_result[i]))
		evaluation1.append(ENS(Data[6], moni_tr_result[i]))
		evaluation1.append(RMSE(Data[6], moni_tr_result[i]))
		evaluation1_all.append(evaluation1)
	
		average_te_elm=[]
		for j in range(NUM):
			average_te_elm.append(((elm_result[i][0][j]).predict(E[i])).flatten())
			
		predict_result=((np.array(average_te_elm)).sum(axis=0))/NUM
		moni_te_result.append(Anti_Normalization(Data[8], predict_result))
		
		evaluation2=[]
		evaluation2.append(r(Data[7], moni_te_result[i]))
		evaluation2.append(ENS(Data[7], moni_te_result[i]))
		evaluation2.append(RMSE(Data[7], moni_te_result[i]))
		evaluation2_all.append(evaluation2)
		
		i=i+1
	
	return A,B, moni_tr_result, evaluation1_all, moni_te_result, evaluation2_all
Evaluation_result1=HPELM_1lead()
