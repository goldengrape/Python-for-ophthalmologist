
# coding: utf-8

# # 变量

# python最简单的功能相当于一个计算器.
# 
# 比如, 眼科最简单的公式SRK公式:
# 
# P = A - 0.9*K - 2.5*AL
# 
# 直接输入数就可以了, 要进行运算时使用shift+return运行. 

# In[12]:


AL=23
A=118.4
K=40
P = A - 0.9*K - 2.5*AL
print(P) 


# 其中AL,A,K,P这些都叫做变量, =符号并不是"等于", 而是被称做赋值的操作. 可以想象变量是一个盒子, 每次可以往里面装不同的东西. 
# 
# 例如, 经常出现这样的问题, 一个病人本来用A常数为118.4的人工晶体, 现在因为某种原因, 突然需要换成A常数为119.0的人工晶体. 
# 
# 变量A可以这样写: 

# In[13]:


A=A+0.6
print(A)


# 如果=符号是"等于", 那么A=A+0.6, 只能推导出A=0, 显然是不对的. 
# 当改变了A常数以后, 如果需要计算新的人工晶体度数

# In[14]:


P = A - 0.9*K - 2.5*AL
print(P)


# ## 练习
# 
# 好了, 该你动手试一试了: 
# * 患者眼轴长24.5mm
# * 角膜曲率: K_1=40, K_2=42
# * 选择使用A常数=119.0 的IOL
# 
# 求目标度数

# In[1]:


# 在此行下面修改代码
AL=None # 将None改成需要的数值
A=None
K_1=None
K_2=None
K=None
P=None # 将None改成需要的计算公式
# 在此行上面修改代码
# 不要改变此行之后的代码
print(P)


# 参考答案: 
#     20.849999999999994

# # 函数

# 如果每次计算都要把P=A - 0.9*K - 2.5*AL写一遍, 是很麻烦的事情, 于是引出一个叫做函数的东西: 
# 函数相当于把常用的计算过程写在一起, 方便反复使用. 

# In[6]:


def SRK(A,K,AL):
    P= A - 0.9*K - 2.5*AL
    return P


# In[7]:


print(SRK(119.0,40,23))


# Python中定义函数的方法是: 
# ```python
# def 函数名(参数1, 参数2, 参数3, 参数4):
#     空4个写函数体的内容
#     return 函数值
# ```
# 
# 比如: def IOL_power(A,K,AL):
# 
# IOL_power就是函数名, A, K, AL就是参数.
# 
# 在召唤函数的时候, 要把赋予参数数值
# 
# IOL_power(119.0,40,23)
# 
# python会按照数字的顺序依次把119.0赋给A, 40赋给K, 23赋给AL

# ## 练习: 
# 重新定义SRK函数, 要使之能够计算两个角膜曲率K_1,K_2的情况. 其中: 
# $$ K=\frac{ K_1+K_2}{2} 
# \\
# P=A - 0.9 \times K + 2.5 \times AL
# $$

# In[9]:


def SRK(A,K_1,K_2,AL):
    # 在此行下面修改代码, 不要改变此行之前的代码
    K=None
    P=None 
    # 在此行上面修改代码, 不要改变此行之后的代码
    return P
print( SRK (119.0, 40, 42, 24.5) )


# 参考答案:  20.849999999999994

# # 分支

# 下面, 我们来做个更复杂一些的SRK-II公式: 
# 
# SRK-II公式中, A常数是根据眼轴长AL改变的: 
# 当
