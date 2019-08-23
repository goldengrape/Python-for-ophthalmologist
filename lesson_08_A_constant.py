#!/usr/bin/env python
# coding: utf-8

# # 人工晶体度数计算中的参数估计与优化
# 
# 发现了一篇[Determination of Personalized IOL-Constants for the Haigis Formula under Consideration of Measurement Precision](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0158988)
# 
# 里面讲的是使用加权最小二乘法来进行Haigis公式的参数优化。我打算使用他的数据，而利用贝叶斯方法做同样的事情。练会这一招以后，其他IOL计算公式的参数优化也是一样的。

# # 获得数据
# 
# 必须得说，临床医学的公开数据比起做计算机的，真是少多了。除了哪些有机器学习强力推动的图像数据集，要找一个开放的数据集还真是困难，虽然理论上，不过就是白内障术前测量和术后复查的IOL数据，好像一抓一大把。但术后IOL稳定至少要3个月，国内的白内障病人3个月后还能追踪到也不是件容易的事情。
# 
# 不过，现在有些杂志要求给出Data Availability，
# 所以加上“Data Availability”这个关键词以后能够找到一些带有原始数据的文章。
# 有些文章假么假事地说are available from the corresponding author on reasonable request.但有些文章很实诚，直接就在Supporting Information里给出excel文件。
# 用google找似乎发现的成功率更高一些。
# 
# 我就是使用这个方法找到了这篇讲 Haigis公式参数优化的文章[Determination of Personalized IOL-Constants for the Haigis Formula under Consideration of Measurement Precision](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0158988)
# 
# 在Supporting Information里面，可以下载到原始的数据。好心的作者已经整理成excel文件了。
# 
# * [数据1](https://doi.org/10.1371/journal.pone.0158988.s001) 植入的是博士伦 EyeCeeOne NS-60YG 人工晶体，一共75只眼。
# * [数据2](https://doi.org/10.1371/journal.pone.0158988.s002) 植入的是Alcon  Acrysof SN60WF 人工晶体，有121只眼
# 
# 我已经把这两个数据都下载下来，放在/data/haigis_data目录中，分别是"S1 File.XLSX", "S1 File.XLSX"

# # 读取数据

# In[1]:


# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(42)
 
# Matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (9, 9)

import seaborn as sns

import os


# In[2]:


# 这里使用了一个小技巧，因为我并不知道您使用的计算机是什么系统的，所以路径使用了os.path.join进行合并。
# pandas可以直接读取excel文件，读取成pandas数据库
s1=pd.read_excel(os.path.join("data","haigis_data","S1 File.XLSX"))
# s2=pd.read_excel(os.path.join("data","haigis_data","S2 File.XLSX"))


# In[3]:


s1.head()


# # 清洗数据
# 
# 读取进来的数据是需要清洗的。excel中有一些是空白的数据，pandas在读取时会以NaN （Not a Number）标记。有些列是不需要的，或者是空列。
# 
# 清洗的时候，先删除掉空列，使用[drop命令](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html), 然后再使用dropna()把含有NaN的行都删掉。注意命令的顺序，如果先删除含有NaN的行，那么就没有剩下的数据了。
# 
# Haigis公式并不针对散光，所以只需要保留其中的眼轴长AL，角膜曲率半径D1，角膜曲率半径D2，前房深度ACD，植入的IOL度数IOL power，和术后6个月的屈光度refraction - 6 months postOP，其他的数据列是不需要的。
# 
# 另外，s1表格中，单位都是以毫米记录的，而在计算过程中通常是以米作为标准单位。所以需要稍微转换一下

# In[4]:


df=pd.DataFrame()
df['AL']=s1['AL'] /1000
df['D1']=s1['D1 [mm]'] /1000
df['D2']=s1['D2 [mm]'] /1000
df['ACD']=s1['ACD'] /1000
df['IOL power']=s1['IOL power']
df["target refraction"]=s1['target refraction']
df['postOP refraction']=s1['refraction - 6 months postOP']
df=df.dropna()


# 简单浏览一下数据的概貌, 丢弃NaN数据以后，保留了69个有效的数据

# In[5]:


df.describe()


# In[6]:


f, axes = plt.subplots(len(df.columns), 1)

for col,ax in zip(df.columns,axes):
    sns.distplot(df[col],ax=ax)


# # Haigis公式
# 
# 这里从文献的引言部分讲起, 请参考[原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0158988#sec001)
# 
# Haigis公式
# $$
# D=\frac{n}{L-d}-\frac{n}{\frac{R_{x}}{K_{m}+\frac{R_{x}}{1-R_{x} \cdot 0.012 m}}-d} \tag{1}
# $$
# 
# 是基于一个简化的角膜薄透镜模型，该模型仅使用角膜前表面的角膜曲率值来计算有效角膜屈光力 Km，使用 nc = 1.332的平均角膜屈光指数进行计算。 
# 
# * Rx是计划屈光度。
# * 术后光学前房深度 d 不一定与术后生理性前房深度相对应。
# * 房水的折射率n = 1.336，
# * ACD是有晶体眼时的前房深度，
# * L是眼轴长度
# 
# **预测**的光学前房深度d，由下面公式给出：
# $$
# d=a_{0}+A C D \cdot a_{1}+L \cdot a_{2} \tag{2}
# $$
# 
# 其中Haigis公式中的a0,a1,a2是IOL型号特异性的，每个厂家每种型号不一定相同。
# 
# 术后**实际**光学前房深度可通过求解方程 1 的 d，推导得出, 将方程 1 中计划的屈光度 Rx 替换为术后实际的屈光度 F。 
# $$
# z=K_{m}+\frac{F}{1-F \cdot 0.012 \mathrm{m}} \tag{3}
# $$
# 
# 得到化简的方程1
# 
# $$
# d=\frac{1}{2}\left(L+\frac{n}{z}\right)-\sqrt{\frac{1}{4}\left(L+\frac{n}{z}\right)^{2}-\left(L \frac{n}{z}-\frac{n^{2}}{D z}+n \frac{L}{D}\right)} \tag{4}
# $$
# 
# 方程4得到的d，应该和方程2得到的d之间误差尽量小，所谓优化参数，就是找到合适的$a_{0},a_{1},a_{2}$, 使得方程4-方程2的平方和最小
# 
# $$
# \chi_{\mathrm{olsq}}^{2}=\sum_{i}\left(a_{0}+A C D_{i} \cdot a_{1}+L_{i} \cdot a_{2}-d_{i}\right)^{2}
# $$
# 
# 

# # 线性回归
# 
# 其实上面就是个线性回归的问题，也就是求解 Y = K X + b
# 
# todo：
# 用PyMC的效果似乎不好，换个优化器，可能lmfit就比较合适，而且通用。

# In[7]:


from lmfit import Minimizer, Parameters, report_fit


# In[8]:


def d_est(n, nc, L,D1,D2,ACD,F,D):
    mean_D=(D1+D2)/2
    Km=(nc-1)/mean_D
    z=Km+F/(1-F*0.012)
    d=1/2*(L+n/z)-np.sqrt(
        1/4*(L+n/z)**2 -
        (L*n/z - n**2/(D*z) + n*L/D)
    )
    return d
df['d']=df.apply(lambda row: 
                 d_est(n=1.336, nc=1.332, L=row.AL, D1=row.D1, D2=row.D2, ACD=row.ACD, F=row["postOP refraction"],D=row['IOL power']),
                 axis=1)


# In[9]:


def fcn2min(params, ACD,L, d):
    """Model a decaying sine wave and subtract data."""
    a0 = params['a0']
    a1 = params['a1']
    a2 = params['a2']
    loss = (a0+ACD*a1+L*a2 -d )**2
    return loss


# create a set of Parameters
params = Parameters()
params.add('a0', value=1.1)
params.add('a1', value=0.4)
params.add('a2', value=0.1)


# In[10]:


# do fit, here with leastsq model
minner = Minimizer(fcn2min, params, fcn_args=(df.ACD,df.AL,df.d))
result = minner.minimize()

# calculate final result
final = df.d + result.residual

# write error report
report_fit(result)


# In[11]:


from IOL_formulas import Haigis
# (R,AC,L,Dl=None, Rx=None, A=None, a0=None,a1=0.400,a2=0.100)


# In[12]:


def fcn2min(params,R, AC, L,Dl, PostOP_Rx):
    """Model a decaying sine wave and subtract data."""
    a0 = params['a0']
    a1 = params['a1']
    a2 = params['a2']
    
    Rx=Haigis(R=R,AC=AC,L=L,Dl=Dl,a0=a0,a1=a1,a2=a2)
    
    
    loss = (Rx-PostOP_Rx )**2
    return loss
# create a set of Parameters
params = Parameters()
params.add('a0', value=1.1)
params.add('a1', value=0.4)
params.add('a2', value=0.1)


# In[15]:


# do fit, here with leastsq model
minner = Minimizer(fcn2min, params, 
                   fcn_args=(
                       (df.D1+df.D2)/2 * 1000,
                       df.ACD * 1000,
                       df.AL * 1000,
                       df["IOL power"],
                       df["postOP refraction"])
                  )
result = minner.minimize()

# write error report
report_fit(result)


# In[16]:


df.columns


# In[27]:


df["opIOL"]=df.apply(lambda row: 
                     Haigis(
                       R=(row.D1+row.D2)/2 * 1000,
                       AC=row.ACD * 1000,
                       L=row.AL * 1000,
                       Rx=row['target refraction'],
                       a0=10.1253329, a1=-0.02132991,a2=-0.23932645),
                     axis=1)
df['opRef']=df.apply(lambda row:
                    Haigis(
                       R=(row.D1+row.D2)/2 * 1000,
                       AC=row.ACD * 1000,
                       L=row.AL * 1000,
                       Dl=row["opIOL"], 
                       a0=10.1253329, a1=-0.02132991,a2=-0.23932645),
                     axis=1)
df['calRef']=df.apply(lambda row:
                    Haigis(
                       R=(row.D1+row.D2)/2 * 1000,
                       AC=row.ACD * 1000,
                       L=row.AL * 1000,
                       Dl=row["IOL power"], 
                       a0=10.1253329, a1=-0.02132991,a2=-0.23932645),
                     axis=1)
                         
                         
#                          R,AC,L,Dl=None, Rx=None, A=None, a0=None,a1=0.400,a2=0.100


# In[28]:


df


# In[20]:


pd.Series(result.residual).hist()


# In[ ]:




