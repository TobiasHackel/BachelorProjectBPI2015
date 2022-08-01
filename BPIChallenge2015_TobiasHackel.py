#!/usr/bin/env python
# coding: utf-8

# In[170]:


import numpy as np
import pandas as pd
import graphviz
import dowhy
from dowhy import CausalModel
from sklearn.preprocessing import LabelEncoder


# In[171]:


path_to_file= "C:/Users/thack/OneDrive/Desktop/BPIC-2015/BPIC15_1_sorted.csv"
df=pd.read_csv(path_to_file,dtype={'dueDate':str,'dateStop':str,'question':str})


# In[172]:


#all closed events
closed=['G']
df=df[df['(case) caseStatus'].isin(closed)]

#only permit denied(geweugerd) or permit issued(verleend)
result=['Vergunning verleend']
df=df[df['(case) last_phase'].isin(result)]

#only Building permits
typ=['Bouw']
df=df[df['(case) parts'].isin(typ)]
df


# In[173]:


#df2=Case-Attribut-Table
df2=df.copy()
df2=df.drop(['Activity','(case) parts','(case) last_phase','Resource','Complete Timestamp','(case) case_type','(case) termName','(case) IDofConceptCase','(case) caseStatus','activityNameNL','concept:name','dateFinished','dateStop','dueDate','lifecycle:transition','planned','(case) caseProcedure','activityNameEN','action_code','monitoringResource','question','(case) landRegisterID','(case) Responsible_actor'],axis=1)
df2=df2.drop_duplicates()
df2


# In[174]:


#convert binary columns to int
lE = LabelEncoder()
l1 = lE.fit_transform(df2['(case) Includes_subCases'])
l1
df2["(case) Includes_subCases"] = l1

lE = LabelEncoder()
l1 = lE.fit_transform(df2['(case) requestComplete'])
l1
df2["(case) requestComplete"] = l1


# In[175]:


#df3= Event-Attribut-Table
df3=df.drop(['Activity','(case) Includes_subCases','monitoringResource','(case) case_type','(case) IDofConceptCase','(case) termName','(case) caseStatus','activityNameNL','concept:name','dateFinished','dateStop','dueDate','lifecycle:transition','planned','(case) termName','(case) caseProcedure','activityNameEN','(case) Responsible_actor','(case) requestComplete','(case) SUMleges','(case) landRegisterID','(case) last_phase','(case) parts','(case) case_type','question'],axis=1)

df3['Complete Timestamp'] = df3['Complete Timestamp'].astype('datetime64')
df3['action_code'] = df3['action_code'].astype('string')

events=['01_HOOFD_010','01_HOOFD_015','01_HOOFD_020','01_HOOFD_065_1','01_HOOFD_065_2','01_HOOFD_110','01_HOOFD_120','01_HOOFD_180','01_HOOFD_200','01_HOOFD_250','01_HOOFD_260','09_AH_I_010','01_HOOFD_370','01_HOOFD_375','01_HOOFD_380','01_HOOFD_430','01_HOOFD_480','01_HOOFD_490','01_HOOFD_490_2']
df3=df3[df3['action_code'].isin(events)]
df3=df3.sort_values(by=['Case ID','Complete Timestamp'])
df3


# In[176]:


#StartTime per case table
event=['01_HOOFD_010']

df4=df3[df3['action_code'].isin(event)]
df4.rename(columns={'Complete Timestamp':'StartTime'}, inplace=True)

df4=df4.drop(['Resource','action_code'],axis=1)
df4


# In[177]:


#Calculate dayspast start to get a int time variable per event
df3=df3.merge(df4,how='left', left_on='Case ID', right_on='Case ID')
for index, row in df3.iterrows():
 df3.at[index,'Complete Timestamp']=(df3.at[(index),'Complete Timestamp']-df3.at[index,'StartTime']).days
df3=df3.drop(['StartTime'],axis=1)
df3.rename(columns={'Complete Timestamp':'dayspast'}, inplace=True)
df3['dayspast']=df3['dayspast'].astype('int')
df3


# In[178]:


#remove event duplicates in case so we can flatten the table without getting column with many NaN values
df3=df3.sort_values(by=['Case ID','action_code','dayspast'])
df3=df3.drop_duplicates(subset=['Case ID','action_code'], keep='last')
df3
    


# In[179]:


#transform action_code to int
lE = LabelEncoder()
l1 = lE.fit_transform(df3['action_code'])
l1
df3["action_code"] = l1
df3


# In[180]:


#flatten table
df3=df3.sort_values(by=['Case ID','dayspast'])
cc = df3.groupby(['Case ID']).cumcount() + 1
df3 = df3.set_index(['Case ID', cc]).unstack().sort_index(1, level=1)
df3.columns = ['_'.join(map(str,i)) for i in df3.columns]
df3.reset_index()
df3


# In[181]:


#merge case table with flattened event table
df3=df3.merge(df2,how='left', left_on='Case ID', right_on='Case ID')
df3=df3.drop(['Case ID'],axis=1)


# In[182]:


df3.rename(columns={'(case) Includes_subCases':'Includes_subCases'}, inplace=True)
df3.rename(columns={'(case) SUMleges':'SUMleges'}, inplace=True)
df3.rename(columns={'(case) requestComplete':'requestComplete'}, inplace=True)


# In[183]:


# We drop AC: 3 R: 10, 13, 14,16 D: 13 14 due to Singular matrix error
df4=df3.drop(['action_code_3','Resource_10','Resource_13','dayspast_13','Resource_14','dayspast_14','Resource_16'],axis=1)

df4


# In[184]:


from causallearn.search.ConstraintBased.PC import (get_adjacancy_matrix,mvpc_alg, pc, pc_alg)
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ConstraintBased.PC import pc
from IPython.display import Image, display
def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)


# In[194]:


cg = pc(df4.to_numpy(),0.05, mv_fisherz, True, 0,0)


# In[186]:


dot = GraphUtils.to_pydot(cg.G, labels=df4.columns)
view_pydot(dot)


# In[188]:


model= dowhy.CausalModel(
        data = df3,
        treatment = "Resource_1+Resource_2+Resource_3+Resource_4+Resource_5+Resource_6+Resource_7+Resource_8+Resource_9+Resource_10+Resource_11+Resource_12+Resource_13+Resource_14+Resource_15+Resource_16+Resource_17+Resource_18".split('+'),
        outcome = "dayspast_18",  
        graph ="digraph{Includes_subCases -> action_code_1;"+
"Includes_subCases -> action_code_10;"+
"Includes_subCases -> action_code_6;"+
"Includes_subCases -> action_code_7;"+
"Includes_subCases -> action_code_8;"+
"Includes_subCases -> action_code_9;"+
"Includes_subCases -> dayspast_6;"+
"Includes_subCases -> dayspast_7;"+
"Includes_subCases -> dayspast_8;"+
"Includes_subCases -> dayspast_9;"+
"Includes_subCases -> dayspast_10;"+
"Resource_1 -> Resource_2;"+
"Resource_1 -> action_code_1;"+
"Resource_1 -> dayspast_1;"+
"Resource_3 -> Resource_4;"+
"Resource_3 -> action_code_3;"+
"Resource_3 -> dayspast_3;"+
"Resource_10 -> Resource_11;"+
"Resource_10 -> action_code_10;"+
"Resource_10 -> dayspast_10;"+
"Resource_13 -> Resource_14;"+
"Resource_13 -> action_code_13;"+
"Resource_13 -> dayspast_13;"+
"Resource_11 -> Resource_12;"+
"Resource_11 -> action_code_11;"+
"Resource_11 -> dayspast_11;"+
"Resource_12 -> Resource_14;"+
"Resource_12 -> action_code_12;"+
"Resource_12 -> action_code_13;"+
"Resource_12 -> dayspast_12;"+
"Resource_14 -> Resource_16;"+
"Resource_14 -> action_code_14;"+
"Resource_14 -> action_code_15;"+
"Resource_14 -> dayspast_14;"+
"Resource_14 -> dayspast_15;"+
"Resource_16 -> Resource_17;"+
"Resource_16 -> action_code_16;"+
"Resource_16 -> dayspast_16;"+
"Resource_17 -> action_code_17;"+
"Resource_17 -> action_code_18;"+
"Resource_17 -> dayspast_17;"+
"Resource_2 -> Resource_4;"+
"Resource_2 -> action_code_2;"+
"Resource_2 -> action_code_3;"+
"Resource_2 -> dayspast_2;"+
"Resource_4 -> Resource_5;"+
"Resource_4 -> action_code_4;"+
"Resource_4 -> dayspast_4;"+
"Resource_5 -> Resource_6;"+
"Resource_5 -> action_code_5;"+
"Resource_5 -> dayspast_5;"+
"Resource_6 -> Resource_7;"+
"Resource_6 -> action_code_6;"+
"Resource_6 -> dayspast_6;"+
"Resource_7 -> Resource_8;"+
"Resource_7 -> action_code_7;"+
"Resource_7 -> dayspast_7;"+
"Resource_8 -> Resource_9;"+
"Resource_8 -> action_code_8;"+
"Resource_8 -> dayspast_8;"+
"Resource_9 -> action_code_10;"+
"Resource_9 -> action_code_9;"+
"Resource_9 -> dayspast_10;"+
"Resource_9 -> dayspast_9;"+
"Resource_15 -> Resource_16;"+
"Resource_15 -> action_code_15;"+
"Resource_18 -> action_code_18;"+
"SUMleges -> action_code_1;"+
"SUMleges -> action_code_10;"+
"SUMleges -> action_code_11;"+
"SUMleges -> action_code_12;"+
"SUMleges -> action_code_13;"+
"SUMleges -> action_code_14;"+
"SUMleges -> action_code_15;"+
"SUMleges -> action_code_16;"+
"SUMleges -> action_code_17;"+
"SUMleges -> action_code_18;"+
"SUMleges -> action_code_2;"+
"SUMleges -> action_code_3;"+
"SUMleges -> action_code_4;"+
"SUMleges -> action_code_5;"+
"SUMleges -> action_code_6;"+
"SUMleges -> action_code_7;"+
"SUMleges -> action_code_8;"+
"SUMleges -> action_code_9;"+
"dayspast_1 -> dayspast_2;"+
"dayspast_1 -> dayspast_3;"+
"dayspast_1 -> dayspast_4;"+
"dayspast_1 -> dayspast_5;"+
"dayspast_2 -> dayspast_6;"+
"dayspast_2 -> dayspast_7;"+
"dayspast_2 -> dayspast_8;"+
"dayspast_3 -> dayspast_6;"+
"dayspast_3 -> dayspast_7;"+
"dayspast_3 -> dayspast_8;"+
"dayspast_4 -> dayspast_6;"+
"dayspast_4 -> dayspast_7;"+
"dayspast_4 -> dayspast_8;"+
"dayspast_5 -> dayspast_6;"+
"dayspast_5 -> dayspast_7;"+
"dayspast_5 -> dayspast_8;"+
"dayspast_6 -> dayspast_9;"+
"dayspast_6 -> dayspast_10;"+
"dayspast_6 -> dayspast_11;"+
"dayspast_6 -> dayspast_12;"+
"dayspast_7 -> dayspast_9;"+
"dayspast_7 -> dayspast_10;"+
"dayspast_7 -> dayspast_11;"+
"dayspast_7 -> dayspast_12;"+
"dayspast_8 -> dayspast_9;"+
"dayspast_8 -> dayspast_10;"+
"dayspast_8 -> dayspast_11;"+
"dayspast_8 -> dayspast_12;"+
"dayspast_9 -> dayspast_15;"+
"dayspast_10 -> dayspast_15;"+
"dayspast_11-> dayspast_15;"+
"dayspast_12 -> dayspast_15;"+
"dayspast_15 -> dayspast_16;"+
"dayspast_15-> dayspast_17;"+
"dayspast_16 -> dayspast_18;"+
"dayspast_17 -> dayspast_18;"+
"requestComplete -> action_code_1;"+
"requestComplete -> action_code_10;"+
"requestComplete -> action_code_11;"+
"requestComplete -> action_code_12;"+
"requestComplete -> action_code_13;"+
"requestComplete -> action_code_14;"+
"requestComplete -> action_code_15;"+
"requestComplete -> action_code_16;"+
"requestComplete -> action_code_17;"+
"requestComplete -> action_code_18;"+
"requestComplete -> action_code_2;"+
"requestComplete -> action_code_3;"+
"requestComplete -> action_code_4;"+
"requestComplete -> action_code_5;"+
"requestComplete -> action_code_6;"+
"requestComplete -> action_code_7;"+
"requestComplete -> action_code_8;"+
"requestComplete -> action_code_9;}"
)
model.view_model()


# In[189]:


identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)


# In[190]:


estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression")
print(estimate)
print("Causal Estimate is " + str(estimate.value))


# In[191]:


refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
print(refute_results)


# In[192]:


estimate = model.estimate_effect(identified_estimand,
        method_name="iv.linear_regression")
print(estimate)
print("Causal Estimate is " + str(estimate.value))


# In[193]:


refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
print(refute_results)


# In[ ]:




