# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:00:13 2016

@author: fox
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

#os.chdir('/home/fox/thesis_code/TRA')
def plot_exp_1():
    file_name='outputs/errors-45corpus-300w2v-LR-maxtime24-05-2016_00:53.csv'
    df= pd.read_csv(file_name,sep=';')

    #read only specified columns and sort in order
    df=df[['Meaning_Error','Sentence_Error','bias_scaling','leak_rate','input_scaling','spectral_radius']]
    df=df.sort_values(['leak_rate','bias_scaling','input_scaling','spectral_radius'])

    # get the unique value from the columns
    iss=np.unique(np.asarray(df["input_scaling"]))
    bs=np.unique(np.asarray(df["bias_scaling"]))
    lr=np.unique(np.asarray(df["leak_rate"]))
    sr=np.unique(np.asarray(df["spectral_radius"]))

    #define ticks and lables for each axes
    xt=range(len(iss))[0:]
    xtl=list(iss)[0:]
    yt=range(len(sr))[0:]
    ytl=list(sr)[0:]

    param_space_dim=[len(lr),len(bs),len(iss),len(sr)]
    print param_space_dim
    #convert error to percentage and reshape with respect to parameter space

    meaning_error=100*np.asarray(df["Meaning_Error"]).reshape(param_space_dim)
    sentence_error=100*np.asarray(df["Sentence_Error"]).reshape(param_space_dim)

    fig=plt.figure(4)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         direction="row",
                         axes_pad=0.5,
                         label_mode="all",
                         share_all="True",
                         cbar_mode="single",
                         cbar_location="right",
                         cbar_size="6%",
                         cbar_pad="2%",
                         )

    # will zip errors wrt first columns of df so that on x-axis will be 2nd columns and y-axis will be 3rd columns
    i=0
    for ax, me in zip(grid, meaning_error[4]):
         ax.set_title('BS: '+str(bs[i]))
         i+=1
         im = ax.imshow(me, cmap = 'seismic',interpolation='nearest',vmin=0, vmax=100,origin='lower', aspect='auto')
         ax.set_xticks(xt)
         ax.set_yticks(yt)
         ax.set_xticklabels(xtl)
         ax.set_yticklabels(ytl)
    fig.colorbar(im)
    plt.show()

if __name__=='__main__':
    plot_exp_1()









