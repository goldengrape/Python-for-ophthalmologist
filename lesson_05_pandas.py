
# coding: utf-8

# # 读取进度
# 
# 玩游戏的话, 如果没有存盘和读取, 简直就没法玩了. 写程序也差不多, 除了要实时保存自己写好的代码, 还需要能够保存和读取数据. 比如做个临床研究, 不大可能手工一个一个按照numpy array的格式录入眼轴长L, 角膜曲率K之类的数据. 这类的数据往往是由机器导出的, 或者由哪个可怜的研究生去病案室抄出来然后存成一个文件的. 
# 
# 这个文件很有可能是excel的. 所以, 这节课我们就以excel为例子读取一组病人的数据, 然后计算IOL度数, 再写入到excel里面. 

# # 文件名
# 
# 欢迎来到现实世界. 前面几课我们面对的都是非常理想化的环境, 我们自己设定什么, 计算机就可以按照我们设想的去做. 但当涉及到文件操作的时候, 就不一定了. 比如, 对于桌面电脑, 可能主流有三类操作系统: Windows, macOS, Linux, 大多数非程序员是用Windows, 但也有相当多的眼科医生用macOS. 在程序员的世界里, 比如我推荐使用的CoCalc或者Azure Notebooks, 其实后台的操作系统是Linux的. 
# 
# 如果你的文件IOLdata.xlsx存储在一个叫做data的文件夹下面, 那么, 问题出现了. 文件名在不同的系统写法会有一点细微的差别: 
# * 对于Windows,       data\IOLdata.xlsx
# * 对于macOS和Linux,  data/IOLdata.xlsx
# 
# 斜杠左右不同, 所以在读取文件的时候, 你不能简单指定文件名是'data\IOLdata.xlsx' 还是 'data/IOLdata.xlsx', 因为你并不知道用程序的人用着什么样的电脑. 
# 
# 萨特说 ** 他人即地狱 **
# 
# 所以我们需要用os库中的os.path.join函数来把文件的路径和文件名按照当前运行时的操作系统要求组合起来.

# In[62]:


import os
pathname='data'
fname='IOLdata.xlsx'
print(os.path.join(pathname,fname))


# 上面的输出结果在不同的电脑上看起来可能会是不同的, 但如果你使用的是[CoCalc](https://cocalc.com)或者[Azure Notebooks](https://notebooks.azure.com)所提供的在线服务, 那么应该显示的是data/IOLdata.xlsx

# # 多个文件名
# 
# 你的合作伙伴很有可能给你的多个文件, 而不仅仅是一个, 当然你可以在excel中依次把它们打开, 然后复制粘贴到一起, (记得把第二个以后文件的标题行删掉). 但如果能够直接用程序读取并且合并不是更好么. 
# 
# 下面, 我要讲解一个之前刻意跳过的内容: List
# 
# 其实也没什么神秘的, 就是用方括号[ ]装起来的一组东西, 这组东西必须是一个类型的, 比如都是数字, 或者都是字符. 
# 如果要访问第0个元素, 就用 list名[0], 第1个元素就用 list名[1]. ** 注意python是从0开始计数的. **

# In[40]:


filenames=['IOLdata00.xlsx','IOLdata01.xlsx']
print(filenames)
print(filenames[0])
print(filenames[1])


# 试试用os.path.join产生一组带有路径的文件名? 

# In[39]:


os.path.join(pathname,filenames)


# 看到出错了吧, 想想你对os.path.join()的要求也太高了, 它怎么知道pathname要跟list里面的每一个元素依次join呢

# # 循环
# 
# 这也是我之前刻意逃避的一个内容, 如果是处理数字, 我推荐尽量避免使用循环, 而直接用向量来处理. 但现在要处理字符串的部分, 可能还是需要介绍一下循环的使用

# In[44]:


filename_list=[]
for f in filenames:
    filename_list.append(os.path.join(pathname,f))
print(filename_list)


# 上面这一段代码中, 我先建立了一个空的list
# ```python
# filename_list=[]
# ```
# 然后用list的一个功能 .append()往里添加元素
# ```python
# filename_list.append(xxxx)
# ```
# 顺便说, x.y()这样的形式, 也是函数, 只不过.y()是x自带的函数, 如果需要知道x都自带了哪些函数, 可以用dir(x)这样的方式查询, 如果需要了解具体那个函数怎么用, 则可以用help. 当然更优选的方案是面向google/stackoverflow的编程.
# ```python
# dir(filename_list)
# help(filename_list.append)
# ```

# 上面的代码中, 还有
# ```python
# for f in filenames:
#     filename_list.append(os.path.join(pathname,f))
# ```
# 这就是一个for循环了. 循环变量f会依次跑遍filenames这个list里面的每一个元素, 然后f携带着list中的元素依次代入到循环体中参与工作. 
# 
# 注意格式, 
# * for语句的尾部需要有冒号: 
# * 循环体要有统一的缩进, 一般是4个空格
# 
# for循环最常见的例子, 恐怕就是从1一直加到100了

# In[59]:


import numpy as np
s=0
for i in np.arange(1,100+1):  # 注意arange也是<, 所以要加到100, arange要到101
   s=s+i
print(s)


# In[56]:


# 我更喜欢这样的方式, 运算速度上会有一点差别, 但这个加法太简单, 不明显
s=np.sum(np.arange(1,101))
print(s)


# 关于循环, 还有下面这样的形式, 
# ```python
# f_list=[os.path.join(pathname,f) for f in filenames]
# ```
# 这句话和
# ```python
# filename_list=[]
# for f in filenames:
#     filename_list.append(os.path.join(pathname,f))
# ```
# 是完全一致的. 写法上简单了很多: 
# * 方括号[ ]表示将产生一个list
# * 第一部分是os.path.join(pathname,f) , 表示list中的元素是怎么来的
# * 第二部分是一个类似for循环的东西, for f in filenames

# In[61]:


f_list=[os.path.join(pathname,f) for f in filenames]
print(f_list)


# 这个叫做list comprehension, 如果看python程序员写的代码, 会有非常多这样的结构出现, 初学者可能认为这样的方式可读性差一些, 属于奇技淫巧, 但由于使用的场景非常多, 所以呈现特殊形式也不难理解. 就像英文里复数并不总加s, 过去式并不总是ed, 而且越是常用的东西往往越不按照语法规则来. 
# 
# 终于收拾好了文件名, 接下来我们可以读取excel文件了

# # pandas

# In[26]:


import pandas as pd # pandas里面还有两个坑, 要确定已经装入了xlrd和openpyxl


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




