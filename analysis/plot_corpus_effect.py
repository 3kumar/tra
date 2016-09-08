# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:09:08 2016

@author: fox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data from csv using pandas
scl_file="../outputs/90k_new/SCL/scl_overall.csv"
sfl_file="../outputs/90k_new/SFL/sfl_overall.csv"

scl_df= pd.read_csv(scl_file,sep=',')
sfl_df= pd.read_csv(sfl_file,sep=',')

#read only specified columns and sort in order
scl_df=scl_df[['Meaning_Error','std. Meaning_Error','Sentence_Error','std. Sentence_Error','corpus']]
sfl_df=sfl_df[['Meaning_Error','std. Meaning_Error','Sentence_Error','std. Sentence_Error','corpus']]

x=range(1,np.asarray(scl_df["corpus"]).shape[0]+1)
x_lbl=np.asarray(scl_df["corpus"])

me_scl = np.asarray(scl_df["Meaning_Error"])
se_scl = np.asarray(scl_df["Sentence_Error"])
me_std_scl=np.asarray(scl_df["std. Meaning_Error"])
se_std_scl=np.asarray(scl_df["std. Sentence_Error"])

me_sfl= np.asarray(sfl_df["Meaning_Error"])
se_sfl = np.asarray(sfl_df["Sentence_Error"])
me_std_sfl=np.asarray(sfl_df["std. Meaning_Error"])
se_std_sfl=np.asarray(sfl_df["std. Sentence_Error"])

# create two subplots for meaning error and sentence error
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2)

# use ax0 to plot meaning error
ax0.errorbar(x, me_scl, yerr=me_std_scl,fmt='g-',label='SCL')
ax0.errorbar(x, me_sfl, yerr=me_std_sfl, fmt='b-',label='SFL')

ax0.set_title('Meaning Error',fontname="Times New Roman Bold",y=1.02,fontsize=18)
ax0.set_xlabel('Sub-Corpora Size (in %)')
ax0.set_ylabel('Cross Validation Error')
ax0.legend()
ax0.set_xlim(0,7)
ax0.set_xticks(x)
ax0.set_xticklabels(x_lbl)
ax0.grid()
ax0.yaxis.set_ticks(np.arange(0.02,0.15,0.01))

#use ax1 to plot sentence error
ax1.errorbar(x, se_scl, yerr=se_std_scl, fmt='g-',label='SCL')
ax1.errorbar(x, se_sfl, yerr=se_std_sfl, fmt='b-',label='SFL')
ax1.set_title('Sentence Error',fontname="Times New Roman Bold",y=1.02,fontsize=18)
ax1.set_xlabel('Sub-Corpora Size (in %)')
ax1.set_ylabel('Cross Validation Error')
ax1.legend()
ax1.grid()
ax1.set_xlim(0,7)
ax1.set_xticks(x)
ax1.set_xticklabels(x_lbl)
ax1.yaxis.set_ticks(np.arange(0.1,0.7,0.05))

plt.tight_layout()
plt.show()
plt.savefig('reservoir_size.png')