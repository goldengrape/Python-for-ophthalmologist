
# coding: utf-8

# # 1001个病人

# In[ ]:


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

