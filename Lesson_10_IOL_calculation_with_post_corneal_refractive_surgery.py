#!/usr/bin/env python
# coding: utf-8

# # 角膜屈光手术后的人工晶体计算
# 
# 参考[Intraocular lens power calculation in eyes with previous corneal refractive surgery](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6053834/)
# 
# 目的：
# * 动手算一遍各种角膜屈光术后的人工晶体计算
# * 顺便练习交互式的Ipython

# In[1]:


import ipywidgets as widgets
from ipywidgets import interact, interact_manual


# In[9]:


def f(x):
    return x


# In[20]:


interact(f, x=widgets.FloatText());


# In[25]:


btn = widgets.Button(description='Medium')
display(btn)
def btn_eventhandler(obj):
    print('Hello from the {} button！'.format(obj.description))
btn.on_click(btn_eventhandler)


# In[ ]:




