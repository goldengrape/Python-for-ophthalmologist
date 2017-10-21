
# coding: utf-8

# # 公式拟合
# 
# 某医院进行临床登记实验, 收集白内障病例2000眼, 进行Phaco+IOL植入术, 使用SRK-2公式进行IOL计算, 
# 
# 术前检查记录: 
# * 选用IOL的A常数: A
# * 角膜曲率: K1, K2
# * 眼轴长: L
# * 目标屈光度: REF
# 
# 术中记录: 
# * 植入IOL度数: P
# 
# 术后3个月复查: 
# * 仅记录了术后对屈光状态不满意, 给予眼镜处方患者的显然验光结果: PostREF
# * 一部分病人失访, 术后验光结果空缺
# 
# 现该医院委托你拟合一个新的IOL计算公式, 所有的病例数据已经整理好存成excel文件, 放在了data文件夹内的sampleIOLdataset.xlsx文件中. 
# 

# # 读取数据集

# In[ ]:


import os
import pandas as pd
pathname='data'
fname='sampleIOLdataset.xlsx'
filename=os.path.join(pathname,fname)
IOLdata=testdata=pd.read_excel(filename)


# In[ ]:


print(IOLdata[0:5])
print(IOLdata.A[0:3])


# # 数据清洗

# In[ ]:


import numpy as np
nanlist=np.isnan(IOLdata.PostOPREF)
print(IOLdata[nanlist][0:4])


# In[ ]:


IOLdata.PostOPREF[nanlist]=IOLdata.REF[nanlist]
# IOLdata=IOLdata.replace(np.nan,0)
print(IOLdata[nanlist][0:4])
y_data=np.asarray(IOLdata.PostOPREF).T.reshape(-1,1)
x_data=np.asarray(IOLdata)[:,0:6]

print(y_data.shape)
print(x_data.shape)


# In[ ]:


y_mean=np.mean(y_data)
y_std=np.std(y_data)
new_y_data=(y_data-y_mean)/y_std
y_train=new_y_data[:1700]
y_test=new_y_data[1700:]


x_mean=np.mean(x_data,axis=0,keepdims=True)
x_std=np.std(x_data,axis=0,keepdims=True)

new_x_data=(x_data-x_mean)/x_std

x_train=new_x_data[:1700]
x_test=new_x_data[1700:]


# # 神经网络简介

# # Keras简介

# https://keras-cn.readthedocs.io/en/latest/

# # 建立深度神经网络

# In[ ]:


# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers,optimizers

model = Sequential()
model.add(Dense(units=64,
                input_dim=6,
                activation="relu",
                kernel_initializer=initializers.glorot_normal(seed=0) ))
model.add(Dense(units=64,
                activation="relu",
                kernel_initializer=initializers.glorot_normal()))
model.add(Dense(units=64,
                activation="relu",
                kernel_initializer=initializers.glorot_normal()))
model.add(Dense(units=64,
                activation="relu",
                kernel_initializer=initializers.glorot_normal()))
model.add(Dense(units=64,
                activation="relu",
                kernel_initializer=initializers.glorot_normal()))
model.add(Dense(units=1,
                activation="tanh",
                kernel_initializer=initializers.glorot_normal()))

from keras.optimizers import SGD
model.compile(loss='mean_squared_error', 
              optimizer='rmsprop',
              #optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])



# In[ ]:


model.fit(x_train, y_train, epochs=200, batch_size=50)


# # 训练拟合

# # 使用深度神经网络进行预测

# In[ ]:




