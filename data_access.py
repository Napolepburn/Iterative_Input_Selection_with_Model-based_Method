from openpyxl import load_workbook
import pandas as pd
import numpy as np
import os

#归一化函数定义
def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min);  
    return x;

#数据读取定义
def read_training_Data():
	#读取盐度数据
	os.chdir('E:\\硕士课题\\近期任务5\\日尺度预测\\数据') #训练时期数据读取
	PG_S_data=pd.read_excel('Salinity_日均盐度.xlsx', 'PG').set_index('Time')
	MK_F_data=pd.read_excel('Sixianjiao Flow_1.xlsx', 'Makou2').set_index('Time')
	SS_F_data=pd.read_excel('Sixianjiao Flow_1.xlsx', 'Sanshui2').set_index('Time')
	HL_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	LL_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	TR_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	W_data=pd.read_excel('Macao Wind_1.xlsx', 'Sheet1').set_index('Time')
	  
	drought_period1A=pd.date_range('2001-10-15','2002-2-28')
	drought_period2A=pd.date_range('2002-10-15','2003-02-28')
	drought_period3A=pd.date_range('2003-10-15','2004-02-28')
	drought_period4A=pd.date_range('2004-10-15','2005-02-28')
	drought_period5A=pd.date_range('2005-10-15','2006-02-28')
	drought_period6A=pd.date_range('2006-10-15','2007-02-28')
	drought_period7A=pd.date_range('2007-11-8','2008-02-28')
	
	drought_period1B=pd.date_range('2001-10-3','2002-2-27')
	drought_period2B=pd.date_range('2002-10-03','2003-02-27')
	drought_period3B=pd.date_range('2003-10-03','2004-02-27')
	drought_period4B=pd.date_range('2004-10-03','2005-02-27')
	drought_period5B=pd.date_range('2005-10-03','2006-02-27')
	drought_period6B=pd.date_range('2006-10-03','2007-02-27')
	drought_period7B=pd.date_range('2007-10-27','2008-02-27')
	
	training_data_time_A=drought_period1A
	training_data_time_A=training_data_time_A.append(drought_period2A)
	training_data_time_A=training_data_time_A.append(drought_period3A)
	training_data_time_A=training_data_time_A.append(drought_period4A)
	training_data_time_A=training_data_time_A.append(drought_period5A)
	training_data_time_A=training_data_time_A.append(drought_period6A)
	training_data_time_A=training_data_time_A.append(drought_period7A)

	training_data_time_B=drought_period1B
	training_data_time_B=training_data_time_B.append(drought_period2B)
	training_data_time_B=training_data_time_B.append(drought_period3B)
	training_data_time_B=training_data_time_B.append(drought_period4B)
	training_data_time_B=training_data_time_B.append(drought_period5B)
	training_data_time_B=training_data_time_B.append(drought_period6B)
	training_data_time_B=training_data_time_B.append(drought_period7B)

	PG_AS_data_o_t1=PG_S_data['PG-S'].ix[training_data_time_A]
	
	PG_AS_data_t1=PG_S_data['PG-S'].ix[training_data_time_B]

	SXJ_F_data_t1=MK_F_data['MK-F'].ix[training_data_time_B]+SS_F_data['SS-F'].ix[training_data_time_B]

	SZ_TR_data_t1=TR_data['TR'].ix[training_data_time_B]

	SZ_HL_data_t1=HL_data['max_l'].ix[training_data_time_B]

	SZ_LL_data_t1=LL_data['min_l'].ix[training_data_time_B]
	
	W_data_t1=W_data['WPS'].ix[training_data_time_B]

	Variable_B1=[]
	Variable_B1.append(PG_AS_data_t1.tolist())
	#Variable_B1.append(PG_ET_data_t1.tolist())
	Variable_B1.append(SXJ_F_data_t1.tolist())
	#Variable_B1.append(GDJ_S_data_t1.tolist())
	Variable_B1.append(SZ_HL_data_t1.tolist())
	Variable_B1.append(SZ_LL_data_t1.tolist())
	Variable_B1.append(SZ_TR_data_t1.tolist())
	Variable_B1.append(W_data_t1.tolist())
	Variable_B1=np.array(Variable_B1)

	L_p1B=len(drought_period1B)
	L_p2B=len(drought_period2B)
	L_p3B=len(drought_period3B)
	L_p4B=len(drought_period4B)
	L_p5B=len(drought_period5B)
	L_p6B=len(drought_period6B)
	L_p7B=len(drought_period7B)
	
	ii=0;F_comb1=[]
	while ii<6:
		lag_v=list(range(1,13))
		jj=0
		while jj<12:
			lag=lag_v[jj]
			a=12-lag;b=L_p1B+1-lag
			c=12-lag+L_p1B;d=L_p1B+L_p2B+1-lag
			e=12-lag+L_p1B+L_p2B;f=L_p1B+L_p2B+L_p3B+1-lag
			g=12-lag+L_p1B+L_p2B+L_p3B;h=L_p1B+L_p2B+L_p3B+L_p4B+1-lag
			i=12-lag+L_p1B+L_p2B+L_p3B+L_p4B;j=L_p1B+L_p2B+L_p3B+L_p4B+L_p5B+1-lag
			k=12-lag+L_p1B+L_p2B+L_p3B+L_p4B+L_p5B;l=L_p1B+L_p2B+L_p3B+L_p4B+L_p5B+L_p6B+1-lag
			m=12-lag+L_p1B+L_p2B+L_p3B+L_p4B+L_p5B+L_p6B;n=L_p1B+L_p2B+L_p3B+L_p4B+L_p5B+L_p6B+L_p7B+1-lag
			
			lag_index=list(range(a,b))+list(range(c,d))+list(range(e,f))+list(range(g,h))+list(range(i,j))+list(range(k,l))+list(range(m,n))
			F_comb1.append(Variable_B1[ii][lag_index]);jj=jj+1
		ii=ii+1

	return PG_AS_data_o_t1,  F_comb1

def read_testing_Data():

	#读取盐度数据
	os.chdir('E:\\硕士课题\\近期任务5\\日尺度预测\\数据')#测试时期数据读取
	PG_S_data=pd.read_excel('Salinity_日均盐度.xlsx', 'PG').set_index('Time')
	MK_F_data=pd.read_excel('Sixianjiao Flow_1.xlsx', 'Makou2').set_index('Time')
	SS_F_data=pd.read_excel('Sixianjiao Flow_1.xlsx', 'Sanshui2').set_index('Time')
	HL_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	LL_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	TR_data=pd.read_excel('SZ-T.xlsx', 'Sheet1').set_index('Time')
	W_data=pd.read_excel('Macao Wind_1.xlsx', 'Sheet1').set_index('Time')

	drought_period7A=pd.date_range('2009-01-01','2009-02-28')
	drought_period8A=pd.date_range('2009-10-15','2010-02-04')

	drought_period7B=pd.date_range('2008-12-20','2009-02-27')
	drought_period8B=pd.date_range('2009-10-03','2010-02-03')

	testing_data_time_A=drought_period7A
	testing_data_time_A=testing_data_time_A.append(drought_period8A)

	testing_data_time_B=drought_period7B
	testing_data_time_B=testing_data_time_B.append(drought_period8B)

	PG_AS_data_o_t2=PG_S_data['PG-S'].ix[testing_data_time_A]#训练对象时间段读取
	PG_AS_data_t2=PG_S_data['PG-S'].ix[testing_data_time_B]
	SXJ_F_data_t2=MK_F_data['MK-F'].ix[testing_data_time_B]+SS_F_data['SS-F'].ix[testing_data_time_B]
	SZ_TR_data_t2=TR_data['TR'].ix[testing_data_time_B]
	SZ_HL_data_t2=HL_data['max_l'].ix[testing_data_time_B]
	SZ_LL_data_t2=LL_data['min_l'].ix[testing_data_time_B]
	W_data_t2=W_data['WPS'].ix[testing_data_time_B]

	Variable_B2=[]
	Variable_B2.append(PG_AS_data_t2.tolist())
	#Variable_B2.append(PG_ET_data_t2.tolist())
	Variable_B2.append(SXJ_F_data_t2.tolist())
	Variable_B2.append(SZ_HL_data_t2.tolist())
	Variable_B2.append(SZ_LL_data_t2.tolist())
	Variable_B2.append(SZ_TR_data_t2.tolist())
	Variable_B2.append(W_data_t2.tolist())
	Variable_B2=np.array(Variable_B2)

	L_p7B=len(drought_period7B)
	L_p8B=len(drought_period8B)

	ll=0;F_comb2=[]
	while ll<6:
		lag_v=list(range(1,13))
		mm=0
		while mm<12:
			lag=lag_v[mm]
			a=12-lag;b=L_p7B+1-lag
			c=12-lag+L_p7B;d=L_p7B+L_p8B+1-lag
			lag_index=list(range(a,b))+list(range(c,d))
			F_comb2.append(Variable_B2[ll][lag_index]);mm=mm+1
		ll=ll+1
		
	return PG_AS_data_o_t2,  F_comb2

def make_PG_Data():
	RESULT_Training_DATA=read_training_Data()
	RESULT_Testing_DATA=read_testing_Data()
	
	RESULT_all_preditors=[]#所有未归一化的预报因子
	for i in range(72):
		RESULT_all_preditors.append(np.concatenate((RESULT_Training_DATA[1][i],RESULT_Testing_DATA[1][i]),axis=0))
	
	RESULT_all_preditors_pre=[]#所有已经归一化的预报因子

	for i in range(int(72/12)):
		transfer=RESULT_all_preditors[12*i].tolist()+RESULT_all_preditors[12*i+1].tolist()+RESULT_all_preditors[12*i+2].tolist()+\
						RESULT_all_preditors[12*i+3].tolist()+RESULT_all_preditors[12*i+4].tolist()+RESULT_all_preditors[12*i+5].tolist()+RESULT_all_preditors[12*i+6].tolist()+\
							RESULT_all_preditors[12*i+7].tolist()+RESULT_all_preditors[12*i+8].tolist()+RESULT_all_preditors[12*i+9].tolist()+RESULT_all_preditors[12*i+10].tolist()+\
								RESULT_all_preditors[12*i+11].tolist()
		for j in range(12):
			RESULT_all_preditors_pre.append(MaxMinNormalization(RESULT_all_preditors[12*i+j],np.array(transfer).max(),np.array(transfer).min()))
	
	
	RESULT_all_preditants=np.concatenate((np.array(RESULT_Training_DATA[0]),np.array(RESULT_Testing_DATA[0])),axis=0)#所有未归一化的预报对象
	
	RESULT_all_preditants_pre=MaxMinNormalization(RESULT_all_preditants,RESULT_all_preditants.max(),RESULT_all_preditants.min())#所有已经归一化的预报对象
	
	
	RESULT_training_preditors_pre=[]#所有已经归一化的用于训练的预报因子
	for i in range(72):
		RESULT_training_preditors_pre.append(RESULT_all_preditors_pre[i][0:935])
	
	
	#所有已经归一化的用于训练的预报对象
	RESULT_training_preditants_pre=RESULT_all_preditants_pre[0:935]
	
	RESULT_testing_preditors_pre=[]#所有已经归一化的用于测试的预报因子
	for i in range(72):
		RESULT_testing_preditors_pre.append(RESULT_all_preditors_pre[i][935:1107])

	#所有已经归一化的用于测试的预报对象
	RESULT_testing_preditants_pre=RESULT_all_preditants_pre[935:1107]
	
	#所有未归一化的用于训练的预报因子
	RESULT_training_preditors=[]
	for i in range(72):
		RESULT_training_preditors.append(RESULT_all_preditors[i][0:935])
		
	#所有未归一化的用于训练的预报对象
	RESULT_training_preditants=RESULT_all_preditants[0:935]
	
	#所有未归一化的用于测试的预报因子
	RESULT_testing_preditors=[]
	for i in range(72):
		RESULT_testing_preditors.append(RESULT_all_preditors[i][935:1107])
		
	#所有未归一化的用于测试的预报对象
	RESULT_testing_preditants=RESULT_all_preditants[935:1107]
	
	#转化成分组形式
	transfer1=[RESULT_training_preditors_pre[0:12], RESULT_training_preditors_pre[12:24], RESULT_training_preditors_pre[24:60], RESULT_training_preditors_pre[60:72]]
	transfer2=[RESULT_testing_preditors_pre[0:12], RESULT_testing_preditors_pre[12:24], RESULT_testing_preditors_pre[24:60], RESULT_testing_preditors_pre[60:72]]
	RESULT_all_preditors_pre1=[RESULT_all_preditors_pre[0:12], RESULT_all_preditors_pre[12:24], RESULT_all_preditors_pre[24:60], RESULT_all_preditors_pre[60:72]]
	
	return  RESULT_all_preditors_pre1, RESULT_all_preditants_pre, transfer1, RESULT_training_preditants_pre, transfer2, RESULT_testing_preditants_pre, RESULT_training_preditants, RESULT_testing_preditants, RESULT_all_preditants

Data=make_PG_Data()


