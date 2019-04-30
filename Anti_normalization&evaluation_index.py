import math
#反归一化
def Anti_Normalization(origin, normalization):
  
    #assert len(origin) == len(normalization)
    Result = []  
    
    for i in range(len(normalization)):
        Result.append(normalization[i]*(origin.max()-origin.min())+origin.min())
    return Result	

#均方根
def RMSE(observed, predicted):
  
    assert len(observed) == len(predicted)
    TRANSFER=0  
    
    for i in range(len(observed)):
    	TRANSFER=TRANSFER+math.pow((observed[i]-predicted[i]),2)
    Result=math.sqrt(TRANSFER/len(observed))
    return Result
	
#纳什效率系数
def ENS(observed, predicted):
  
    assert len(observed) == len(predicted)
    TRANSFER1=0;TRANSFER2=0;TRANSFER3=sum(observed)/len(observed)
    
    for i in range(len(observed)):
    	TRANSFER1=TRANSFER1+math.pow((observed[i]-predicted[i]),2)
		
    for j in range(len(observed)):
    	TRANSFER2=TRANSFER2+math.pow((observed[j]-TRANSFER3),2)
		
    Result=1-TRANSFER1/TRANSFER2
    return Result
	
#相关系数
def r(observed, predicted):
  
    assert len(observed) == len(predicted)
    TRANSFER1=0;TRANSFER2a=0;TRANSFER2b=0;TRANSFER3=sum(observed)/len(observed);TRANSFER4=sum(predicted)/len(predicted)
    
    for i in range(len(observed)):
    	TRANSFER1=TRANSFER1+(observed[i]-TRANSFER3)*(predicted[i]-TRANSFER4)
		
    for j in range(len(observed)):
    	TRANSFER2a=TRANSFER2a+math.pow((observed[j]-TRANSFER3),2)
		
    for k in range(len(observed)):
    	TRANSFER2b=TRANSFER2b+math.pow((predicted[k]-TRANSFER4),2)	
		
    Result=TRANSFER1/(math.sqrt(TRANSFER2a)*math.sqrt(TRANSFER2b))
    return Result
