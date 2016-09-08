# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:09:08 2016

@author: fox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data from csv using pandas
scl_file="../outputs/corpus462/start/reservoir_size-tra-462-1000res-10folds-1e-06ridge-50w2vdim-<start>-04-09_02:01.csv"
sfl_file="../outputs/corpus462/end/reservoir_size_tra-462-1000res-10folds-1e-06ridge-50w2vdim-<end>-04-09_02:15.csv"

scl_df= pd.read_csv(scl_file,sep=';')
sfl_df= pd.read_csv(sfl_file,sep=';')

#****************************************************************************************************
# read only specified columns and sort in order
scl_df=scl_df[['Meaning_Error','Sentence_Error','reservoir_size','seed']]

#get the mean and standard deviation in meaning and sentence errors for all reservoir size by averaging over all instances
scl_df_mean=scl_df.groupby(['reservoir_size'])['Meaning_Error','Sentence_Error'].mean().reset_index()
scl_df_std=scl_df.groupby(['reservoir_size'])['Meaning_Error','Sentence_Error'].std().reset_index()


#*****************************************************************************************************
# read only specified columns and sort in order
sfl_df=sfl_df[['Meaning_Error','Sentence_Error','reservoir_size','seed']]

#get the mean and standard deviation in meaning and sentence errors for all reservoir size by averaging over all instances
sfl_df_mean=sfl_df.groupby(['reservoir_size'])['Meaning_Error','Sentence_Error'].mean().reset_index()
sfl_df_std=sfl_df.groupby(['reservoir_size'])['Meaning_Error','Sentence_Error'].std().reset_index()

#*******************************************************************************************************

x=range(1,np.asarray(scl_df_mean["reservoir_size"]).shape[0]+1)
x_lbl=np.asarray(scl_df_mean["reservoir_size"])

me_scl = np.asarray(scl_df_mean["Meaning_Error"])
se_scl = np.asarray(scl_df_mean["Sentence_Error"])
me_std_scl=np.asarray(scl_df_std["Meaning_Error"])
se_std_scl=np.asarray(scl_df_std["Sentence_Error"])

me_sfl= np.asarray(sfl_df_mean["Meaning_Error"])
se_sfl = np.asarray(sfl_df_mean["Sentence_Error"])
me_std_sfl=np.asarray(sfl_df_std["Meaning_Error"])
se_std_sfl=np.asarray(sfl_df_std["Sentence_Error"])

# create two subplots for meaning error and sentence error
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2)

# use ax0 to plot meaning error
ax0.errorbar(x, me_scl, yerr=me_std_scl, fmt='r-',label='SCL')
ax0.errorbar(x, me_sfl, yerr=me_std_sfl, fmt='b-',label='SFL')

ax0.set_title('Meaning Error',fontname="Times New Roman Bold")
ax0.set_xlabel('Reservoir Size')
ax0.set_ylabel('Cross validation errors')
ax0.legend()
ax0.grid()
ax0.set_xlim(0,11)
ax0.set_xticks(x)
ax0.set_xticklabels(x_lbl)
ax0.yaxis.set_ticks(np.arange(0.0,0.65,0.05))

#use ax1 to plot sentence error
ax1.errorbar(x, se_scl, yerr=se_std_scl, fmt='r-',label='SCL')
ax1.errorbar(x, se_sfl, yerr=se_std_sfl, fmt='b-',label='SFL')
ax1.set_title('Sentence Error',fontname="Times New Roman Bold")
ax1.set_xlabel('Reservoir Size')
ax1.set_ylabel('Cross validation errors')
ax1.legend()
ax1.grid()
ax1.set_xlim(0,11)
ax1.set_xticks(x)
ax1.set_xticklabels(x_lbl)

ax1.yaxis.set_ticks(np.arange(0.1,1.2,0.1))

plt.tight_layout()
plt.show()
plt.savefig('reservoir_size.png')