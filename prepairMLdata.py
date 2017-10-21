
# coding: utf-8

# # 生成随机数据
# 
# 利用SRK/T公式, 产生大量的IOL数据, 用来进行机器学习训练用. 
# 如果是临床病人, 那么术前应该取得的数据有: 
# * A常数
# * 角膜曲率K1, K2
# * 眼轴长L
# * 目标屈光度REF
# 
# 术中假定按照SRK-II植入了IOL
# * IOL实际植入数据P
# 
# 术后3个月, 屈光稳定, 重新验光
# * 术后屈光度R, 用SRKT公式算出Pr-P+随机误差
# * 但有很多病人其实术后没什么不满意, 于是失访了, 我们假定术后验光<=0.75D就不来了, 我们只好假定验光为正视眼, 或者正视眼医生也没留下处方记录. 
# * 还有一部分是随机失访
# 

# In[4]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from IOLfomular import testdata,SRK_2,SRK_T


# In[7]:



def generate_dataset(population):
    preOPdata=testdata(population)
    A=np.asarray(preOPdata['A'])
    K1=np.asarray(preOPdata['K1'])
    K2=np.asarray(preOPdata['K2'])
    L=np.asarray(preOPdata['L'])
    REF=np.asarray(preOPdata['REF']) 
    
    P=SRK_2(A,K1,K2,L,REF)
#     P=np.around(P*4)/4
    Pr=SRK_T(A,K1,K2,L,REF)         
    
    noise=np.random.randn(population,1)*0.0
    ratio=np.random.rand(population,1)*0.0+0.8
    R=ratio*(P-Pr)+noise
#     R=np.around(R*4)/4
#     satisfied=np.logical_and(R<0,R>-0.50)
#     loss_to_follow=np.random.rand(population,1)<0.05
#     R[satisfied]=0
#     R[loss_to_follow]=np.nan
    data=np.asarray([preOPdata['A'],
            preOPdata['K1'],
            preOPdata['K2'],
            preOPdata['L'],
            preOPdata['REF'],
            P,
            R]).reshape(7,population).T
    result=DataFrame(data, columns=['A','K1','K2','L','REF','IOLPower','PostOPREF']
            )
    return result

dataset=generate_dataset(2000)


# # 保存

# In[8]:


pathname='data'
filename='sampleIOLdataset.xlsx'
dataset.to_excel(os.path.join(pathname,filename))


# In[ ]:




