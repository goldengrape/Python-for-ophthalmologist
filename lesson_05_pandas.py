
# coding: utf-8

# In[26]:


import pandas as pd # pandas里面还有两个坑, 要确定已经装入了xlrd和openpyxl
import numpy as np
import os 


# In[27]:


pathname='data'
filename=['IOLdata00.xlsx','IOLdata01.xlsx']
print(os.path.join(pathname,filename[0]))
filename_list=[]
for f in filename:
    filename_list.append(os.path.join(pathname,f))
print(filename_list)
f_list=[os.path.join(pathname,f) for f in filename]
print(f_list)


# In[28]:


IOLdata_list=[pd.read_excel(f) for f in f_list]
IOLdata=pd.concat(IOLdata_list,ignore_index=True)
print(IOLdata[17:21])


# In[29]:


print(IOLdata.K1[0:5])


# In[30]:


def on_L_change_A(L,A,Lmin,Lmax,deltaA):
    pickout=np.logical_and(L>Lmin, L<=Lmax)
    A[pickout] += deltaA
    return A
def SRK_2(A,K_1,K_2,L,REF=0):
    A = np.asarray(A).copy() # 避免pandas修改原始数据, 还有更好的方案么? 
    A = on_L_change_A(L,A,0,    20,    3)
    A = on_L_change_A(L,A,20,    21,    2)
    A = on_L_change_A(L,A,21,    22,    1)
    A = on_L_change_A(L,A,22,    24.5,  0)
    A = on_L_change_A(L,A,24.5, 50,    -0.5)

    K = (K_1+K_2)/2
    P_emme= A - 0.9*K -2.5*L
    CR = np.ones(P_emme.shape)
    CR[ P_emme>=14 ]=1.25
        
    P_ammc=P_emme-REF*CR
    return P_ammc


# In[31]:


IOLpower=SRK_2(IOLdata.A, IOLdata.K1,IOLdata.K2,IOLdata.AL)


# In[32]:


newIOLdata=pd.concat([IOLdata,IOLpower],axis=1)


# In[33]:


#print(newIOLdata)
# 要确保装入了openpyxl
newIOLdata.to_excel(os.path.join(pathname,'output.xlsx'))


# In[ ]:




