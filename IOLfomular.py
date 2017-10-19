
# coding: utf-8

# # IOL 公式集
# 
# 这里我尽量把现有的IOL计算公式都用python写出来, 由于我使用了jupyter的同步存储插件, 所以所有的代码也会同步保存为IOLfomular.py, 因此可以很方便在其他的python程序中以import IOLfomular的方式进行调用. 
# 
# 这些计算公式的函数可能有多个版本, 以适应对单一病人数据求解的情况和对批量病人数据求解的情况. 
# 
# 大多数运算使用numpy会更加简单和迅速, 所以要首先

# In[27]:


import numpy as np


# In[101]:


# 生成标准测试集
if __name__ == "__main__" :
    np.random.seed(0)
IOL_A_Const={
    # 大概需要用个JSON文件存一下, 数据太多了
    'AcrySof IQ': 118.7,
    'AcrySof IQ Toric': 119.0,
    'PCB00': 118.8
}

def testdata(patinets_num):
    possibleA=list(IOL_A_Const.values())
    A=np.random.choice(possibleA,patinets_num,1).reshape(patinets_num,1)
    K1=np.random.rand(patinets_num,1)*2+40
    K2=np.random.rand(patinets_num,1)*2+40
    L=np.random.rand(patinets_num,1)*3+23
    patient_data={
        'A': A,
        'K1': K1,
        'K2': K2,
        'L': L
    }
    return patient_data
    


# In[116]:


data=testdata(4)
data['L']


# # SRK
# 
# $$
# P= A - 0.9 \times K -2.5 \times L
# $$
# 其中A是
# * A常数, 
# * K是平均角膜曲率, $ K=\frac{K_1+K_2}{2} $
# * L是眼轴长
# 
# SRK 公式比较简单, 并没有复杂的判断过程. 可以同时适用于单一病人和批量病人的求解

# In[103]:


def SRK(A, K1, K2,L):
    K=(K1+K2)/2
    P= A - 0.9*K - 2.5*L
    return P


# In[104]:


if __name__ == "__main__" :
    pNum=[1,4]
    for p in pNum:
        data=testdata(p)
        print(SRK(data['A'],
            data['K1'],
            data['K2'],
            data['L']
           ))


# # SRK-II
# 
# SRK-II公式, 这个公式中经过修正的A常数A1, 是和眼轴长L有关的. 
# 
# * 当 L< 20.0 时, A1= A+3
# * 当 20.0<= L < 21.0 时, A1= A+2
# * 当 21.0<= L < 22.0 时, A1= A+1
# * 当 22.0<= L < 24.5 时, A1= A
# * 当 L > 24.5.0 时, A1= A-0.5
# 
# $$
# P= A1 - 0.9 \times K -2.5 \times L
# $$
# 
# 为了支持多个病人的数据以向量的方式输入, 就不能简单使用if来做判断. 

# In[105]:


def on_1st_change_2nd(L,A,Lmin,Lmax,deltaA):
    if not(np.isscalar(L)):
        assert A.shape==L.shape
    pickout=np.logical_and(L>Lmin, L<=Lmax)
    A[pickout] += deltaA
    return A
def SRK_2(A,K_1,K_2,L,REF=0):
    A = np.asarray(A).copy() # 避免pandas修改原始数据, 还有更好的方案么? 
    A = on_1st_change_2nd(L,A,0,     20,    3)
    A = on_1st_change_2nd(L,A,20,    21,    2)
    A = on_1st_change_2nd(L,A,21,    22,    1)
    A = on_1st_change_2nd(L,A,22,    24.5,  0)
    A = on_1st_change_2nd(L,A,24.5,  50,    -0.5)

    K = (K_1+K_2)/2
    P_emme= A - 0.9*K -2.5*L
    CR = np.ones(P_emme.shape)
    CR[ P_emme>=14 ]=1.25
        
    P_ammc=P_emme-REF*CR
    return P_ammc


# In[106]:


if __name__ == "__main__" :
    pNum=[1,4]
    for p in pNum:
        data=testdata(p)
        print(SRK_2(data['A'],
            data['K1'],
            data['K2'],
            data['L']
           ))


# # SRK-T
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

# In[158]:


def SRK_T(A,K1,K2,L, REF=0): 
    V=12
    na=1.336
    nc=1.333
    K=(K1+K2)/2
    r = 337.5/K
    L=np.asarray(L)
    LC=L.copy()    
    mLlist=L>24.2
    LC[mLlist]=-3.446+1.716*L[mLlist]-0.0237*(L[mLlist]**2)
    
    W = -5.41+0.58412*LC + 0.098*K
    print("W")
    print(W)

    H = r-np.sqrt(r**2-(W**2)/4)
    Ofst = (0.62467*A-68.747)-3.336
    C1 = H + Ofst
    L1 = L+(0.65696- 0.02029 * L)
    X = na*r-L1*(nc-1)
    Y = na*r - C1*(nc -1)
    
    P_emme = ( 1000*na*X ) / ((L1-C1) * Y)
    P_amet = ( 1000*na*(X-0.001*REF*(V*X+L1*r) )) / ((L1-C1)*(Y-0.001*REF*(V*X+C1*r)))   

    # 此行以上填写
    return P_amet


# In[159]:


if __name__ == "__main__" :
    pNum=[1,2]
    for p in pNum:
        data=testdata(p)
        print(p)
        print(SRK_T(data['A'],
            data['K1'],
            data['K2'],
            data['L']
           ))


# In[ ]:




