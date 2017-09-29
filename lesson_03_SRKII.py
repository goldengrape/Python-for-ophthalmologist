
# coding: utf-8

# # 条件判断
# 
# 来看SRK-II公式, 这个公式中经过修正的A常数A1, 是和眼轴长L有关的. 
# 
# * 当 L< 20.0 时, A1= A+3
# * 当 20.0<= L < 21.0 时, A1= A+2
# * 当 21.0<= L < 22.0 时, A1= A+1
# * 当 22.0<= L < 24.5 时, A1= A
# * 当 L > 24.5.0 时, A1= A-0.5
# 
# 这是一个分段函数, 需要判断条件, 才能决定另一个参数. 遇到这种情况就需要使用if语句了. 
# ``` python
# if 条件1为真 :
#     执行这一段代码
# elif 条件2为真 :
#     执行这一段代码
# else : 
#     以上条件都不满足, 执行这一段代码
# ```
# 
# 不知道有人注意过没有, 在医学公式里, 存在条件判断是常态. 而在物理学里面, 特别是描述世界本质的理论物理, 据我所知并没有一个是带有if语句的公式. 真是奇怪的世界. 
# 

# In[1]:


def SRK_2(A,K_1,K_2,L,REF=0):
    if L < 20.0 :
        A1=A+3
    elif L < 21.0 :
        A1=A+2
    elif L < 22.0 :
        A1=A+1
    elif L == 22.0:
        A1=A
    elif L > 22.0 and L < 24.5 :
        A1=A
    elif L >= 24.5:
        A1=A-0.5
    
    K = (K_1+K_2)/2
    P_emme= A1- 0.9*K -2.5*L
    
    if P_emme < 14:
        CR = 1.00
    else:
        CR = 1.25
        
    P_ammc=P_emme-REF*CR
    return P_ammc


# In[2]:


print(SRK_2(118.4,40,42,22.2))


# 注意一些if的使用细节: 
# 
# * 每个if/ elif/ else后面都要有冒号: 
# * 每个代码块要注意有相同的缩进
# * python用两个等号==表示等于
# * 如果需要判断多个逻辑关系, 可以用and 或者 or 来将各个判断条件联系起来. 医学文献里经常出现的and/or是非常错误的写法. 
# 
# 在上面的函数定义中, 除了用到了if elif else以外, 还有一个小细节: 
# ```python
# def SRK_2(A,K_1,K_2,L,REF=0):
# ```
# 其中REF=0的含义是, 这个REF参数有缺省值=0, 如果在调用函数的时候没有写它, 这个参数就等于缺省值. 
# 

# In[3]:


print("术后预留-3D: ")
print(SRK_2(118.4,40,42,22.2, REF=-3.00))
print(SRK_2(118.4,40,42,22.2,-3.00))
print("术后保留正视眼: ")
print(SRK_2(118.4,40, 42,22.2))


# # 练习
# 干脆把几个IOL公式都写了吧: 
# 
# 
# 

# ## SRK-T
# * 正视眼: 
# P_emme = ( 1000 * na * X ) / ((L1-C1) * Y) 
# 
# * 屈光不正眼: 
# P_amet = ( 1000 * na * (X-0.001*REF*(V*X+L1*r) ))/((L1-C1)*(Y-0.001*REF*(V*X+C1*r))
# 
# * 其中: 
#   * X  = na*r-L1*(nc-1)
#   * Y  = na*r - C1*(nc -1)
#   * L1 光学视轴长
#     * L1=L+(0.65696- 0.02029 * L)
#   * REF 目标屈光度
#   * r 平均角膜曲率
#     * r = 337.5/K
#   * W 计算角膜厚度
#     * W= -5.41+0.58412 * LC + 0.098 * K
#   * LC 修正眼轴长
#     * if L<=24.2: LC=L
#     * if L>24.2:  LC=-3.446+1.176*L-0.237*(L**2)
#   * C1 估计术后前房深度
#     * C1=H + Ofst
#   * Ofst Calculated distance between  the iris sufrace and IOL optical surface including corneal thickness(mm)
#     * Ofst=(0.62467*A-68.747)-3.336
#   * H 角膜穹顶高
#     * $ H=r-\sqrt{r^2-W^2/4} $
#   * A : A常数
#   * K 平均角膜屈光度
#     * K=(K1+K2)/2
#   * P 植入IOL度数
#   * V 顶点距离 V=12
#   * na 房水和玻璃体折射率 na=1.336
#   * nc 角膜折射率 nc=1.333
#     
# 洋人非常喜欢倒叙, 注意在写程序的时候要把叙述的顺序搞清楚. 

# ## 导入数学库
# 
# $$ H=r-\sqrt{r^2-W^2/4} $$
# 
# 需要注意的是, 在计算H的时候, 用到了平方和开方. 
# 
# 平方用两个 * 号表示: $r^2==r**2 $
# 
# python的标准库里是没有开方运算的. 需要导入一个数学库numpy, 今后我们会经常使用numpy
# 
# 比如计算 $ \sqrt{2} $

# In[4]:


import numpy as np
np.sqrt(2)


# ```python
# import numpy as np
# ```
# 这句话的意思是, 导入numpy这个数学库, 并且把它简称为np, 虽然也可以简称为其他, 但大家习惯上还是使用np. 
# 
# 导入了数学库以后, 要调用数学库中的运算, 就用np.XXX( ), 比如开方:  np.sqrt(2)
# 
# 通常来说导入外部库的import语句放在程序的最开头. 不过反正在调用之前都可以, 而且只要import一次就好了. 
# 
# 

# In[5]:


def SRK_T( REF=0): # 参数表会有不少哦, 自己看着填吧
    V=12
    na=1.336
    nc=1.333
    K=(K1+K2)/2
    r = 337.5/K
    # 此行以下填写
    if L<=24.2: 
        LC=None
    elif L>24.2: 
        LC=None
    
    W = None
    H = r-np.sqrt(r**2-(W**2)/4)
    Ofst = None
    C1 = None
    L1 = None
    X = None
    Y = None
    
    P_amet = None

    # 此行以上填写
    return P_amet


# In[ ]:




