# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:09:08 2016

@author: fox

This script is used to plot the effect of corpus size on the TRA task. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pgf_with_latex = {
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif":[],
        'font.size': 16,                   # blank entries should cause plots to inherit fonts from the document
        'axes.titlesize':'medium',
        "axes.labelsize": 'medium',
        "legend.fontsize": 12,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        }

plt.rcParams.update(pgf_with_latex)

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

print '\nSCL mode\n'
print me_scl,me_std_scl
print se_scl,se_std_scl


me_sfl= np.asarray(sfl_df["Meaning_Error"])
se_sfl = np.asarray(sfl_df["Sentence_Error"])
me_std_sfl=np.asarray(sfl_df["std. Meaning_Error"])
se_std_sfl=np.asarray(sfl_df["std. Sentence_Error"])

print '\nSFL mode\n'
print me_sfl,me_std_sfl
print se_sfl,se_std_sfl
# create two subplots for meaning error and sentence error
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2)

# use ax0 to plot meaning error
ax0.errorbar(x, me_scl, yerr=me_std_scl,fmt='g-',label='SCL')
ax0.errorbar(x, me_sfl, yerr=me_std_sfl, fmt='b-',label='SFL')

ax0.set_title(r'\textbf{Meaning Error}',y=1.02)
ax0.set_xlabel('Sub-corpora size (in \%)}')
ax0.set_ylabel('Cross validation error')
ax0.legend()
ax0.set_xlim(0,7)
ax0.set_xticks(x)
ax0.set_xticklabels(x_lbl)
ax0.grid(alpha=0.6,c='grey',ls='dotted')
ax0.yaxis.set_ticks(np.arange(0.0,0.16,0.02))

#use ax1 to plot sentence error
ax1.errorbar(x, se_scl, yerr=se_std_scl, fmt='g-',label='SCL')
ax1.errorbar(x, se_sfl, yerr=se_std_sfl, fmt='b-',label='SFL')
ax1.set_title(r'\textbf{Sentence Error}',y=1.02)
ax1.set_xlabel('Sub-corpora size (in \%)')
ax1.set_ylabel('Cross validation error')
ax1.legend()
ax1.grid(alpha=0.6,c='grey',ls='dotted')
ax1.set_xlim(0,7)
ax1.set_xticks(x)
ax1.set_xticklabels(x_lbl)
ax1.yaxis.set_ticks(np.arange(0.1,0.7,0.1))

plt.show()
plt.tight_layout()
fig.set_size_inches(10, 5.5)
plt.savefig('/home/fox/thesis_report/src/corpus_size_1_test.pdf')