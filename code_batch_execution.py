import os
import math
import numpy as np
from pandas import Series
from scipy import stats
import matplotlib.pyplot as plt

#读取的文件路径(支持批量读取)
read_path=['../数据读取与预处理\\b2-PG-DAS数据读取_1d（归一化）.py',\
				'../数据读取与预处理\\b2-PG-DAS数据读取_2d（归一化）.py',\
					'../数据读取与预处理\\b2-PG-DAS数据读取_3d（归一化）.py']

#可视化保存的文件路径(支持批量保存)
visualization_path=['../每一轮的结果示意图\\b2-1d.png',\
						'../每一轮的结果示意图\\b2-2d.png',\
							'../每一轮的结果示意图\\b2-3d.png']
							
benchmark_path=os.path.abspath('.')   # 定义当前所处的文件夹的绝对路径

#定义可视化程序
def Visualization(save_path):
	#整理数据
	Performance_indicies_tr_R=[]
	Performance_indicies_tr_NSE=[]
	Performance_indicies_tr_RMSE=[]
	for i in range(len(Evaluation_result1[3])):
		Performance_indicies_tr_R.append(Evaluation_result1[3][i][0])
		Performance_indicies_tr_NSE.append(Evaluation_result1[3][i][1])
		Performance_indicies_tr_RMSE.append(Evaluation_result1[3][i][2])
	Performance_indicies_tr=[Performance_indicies_tr_R,Performance_indicies_tr_NSE,Performance_indicies_tr_RMSE]

	
	Performance_indicies_te_R=[]
	Performance_indicies_te_NSE=[]
	Performance_indicies_te_RMSE=[]
	for i in range(len(Evaluation_result1[5])):
		Performance_indicies_te_R.append(Evaluation_result1[5][i][0])
		Performance_indicies_te_NSE.append(Evaluation_result1[5][i][1])
		Performance_indicies_te_RMSE.append(Evaluation_result1[5][i][2])
	Performance_indicies_te=[Performance_indicies_te_R,Performance_indicies_te_NSE,Performance_indicies_te_RMSE]
		
	A=len(Selection_result[0][0])
	
	round_num=len(Performance_indicies_tr_R)
		
	#可视化
	label=["Training_R","Testing_R","Training_NSE","Testing_NSE"]

	x=np.linspace(0,round_num-1,round_num)
	plt.figure(figsize=(6,4))
	plt.plot(x,Performance_indicies_tr_R,label=label[0],linewidth=0.5,marker='*',color='blue')
	plt.plot(x,Performance_indicies_te_R,label=label[1],linestyle='-.',linewidth=0.5,marker='*',color='red')
	plt.plot(x,Performance_indicies_tr_NSE,label=label[2],linewidth=0.5,marker='v',color='blue')
	plt.plot(x,Performance_indicies_te_NSE,label=label[3],linestyle='-.',linewidth=0.5,marker='v',color='red')
	# 增加竖线
	plt.axvline(x=A-0.5, linestyle='--',color='black', linewidth=1)
	
	plt.xlabel("Iterative_round")
	plt.ylabel("Performance_indicies_values")
	#plt.grid(alpha=0.5)
	plt.legend(loc='SouthEast', fontsize=10)  # 图像大小
	#plt.plot([x0, x0,],[0,y0,], 'k--',linewidth=1.0)
	plt.savefig(save_path,dpi=800,bbox_inches='tight') # 图像输出分辨率
	plt.show()
	
	return Performance_indicies_tr, Performance_indicies_te
	
#批量处理数目
batch_num=len(read_path)

i=0
while i<batch_num:

	#读取数据
	os.chdir(benchmark_path)
	with open(read_path[i],'r',encoding='UTF8') as f:
		exec(f.read())
	#读取反归一化模块
	os.chdir(benchmark_path)
	with open('../执行程序\\反归一化与评价指标模块.py','r',encoding='UTF8') as f:
		exec(f.read())
	#因子挑选
	os.chdir(benchmark_path)
	with open('../执行程序\\OP-HPELM因子挑选模块.py','r',encoding='UTF8') as f:
		exec(f.read())
	#ELM运行
	os.chdir(benchmark_path)
	with open('../执行程序\\ELM自动模拟程序.py','r',encoding='UTF8') as f:
		exec(f.read())
	#可视化及保存
	os.chdir(benchmark_path)
	Visual_result=Visualization(visualization_path[i])
	
	i+=1

