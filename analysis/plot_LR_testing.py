# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:17:45 2016

@author: fox
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_errors(csv_file_name=None,rows=2,cols=2, plot_for='leak_rate', x='spectral_radius', y='input_scaling', z='Sentence_Error', average_over='seed'):
    '''
        this methods create plots as many input_scaling values are there.
        each plot is an image graph of grids , where on y_axis is spectral_radius
        and on x_axis is leak_rate.

    '''
    if csv_file_name is None:
        csv_file='../outputs/param-search-tra-462-1000res-10folds-1e-06ridge-50w2vdim-<start>-13-09_23:05.csv'
    else:
        csv_file=csv_file_name

    df= pd.read_csv(csv_file,sep=';')

    #read only specified columns and sort in order NOTE: Do not change y, x order

    df = df[[average_over, x, y, z]].sort()
    df_g=df.groupby([x,y,z]).mean().reset_index()

    #df_g.to_csv('../outputs/new_search/averaged_start_broad.csv')

    print df_g

    '''
    ax_title=np.unique(np.asarray(df[plot_for]))
    x_labels=np.unique(np.asarray(df[x]))
    y_labels=np.unique(np.asarray(df[y]))

    # this will be used later to form grids of SR * LR for all input scaling
    #param_space_dim=[len(self.iss),len(self.sr),len(self.lr)]
    param_space_dim=[len(ax_title),len(y_labels),len(x_labels)]

    yt=range(len(y_labels))
    ytl=list(y_labels)

    xt=range(len(x_labels))
    xtl=list(x_labels)

    #convert error to percentage and reshape with respect to parameter space
    #me=100*np.asarray(self.df_me["Meaning_Error"]).reshape(param_space_dim)
    me=np.asarray(df_g[z]).reshape(param_space_dim)

    len(ax_title)/2

    fig=plt.figure(7)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(rows,cols),
                         direction="row",
                         axes_pad=0.8,
                         add_all=True,
                         label_mode="all",
                         share_all="True",
                         cbar_mode="single",
                         cbar_location="right",
                         cbar_size="6%",
                         cbar_pad="2%",
                         )

    # will zip errors wrt first columns of df so that we get grid for 2nd and 3rd axis
    i=0
    for ax, me in zip(grid, me):
         ax.set_title(plot_for+':'+str(ax_title[i]))
         i+=1
         im = ax.imshow(me,cmap = 'jet', interpolation='nearest', vmin=0, vmax=1,origin='lower', aspect='auto')
         ax.set_xticks(xt)
         ax.set_xticklabels(xtl)
         ax.set_xlabel(x)

         ax.set_yticks(yt)
         ax.set_yticklabels(ytl)
         ax.set_ylabel(y)

    ax.cax.colorbar(im)
    plt.tight_layout()
    plt.show()'''

if __name__=='__main__':
    #csv_file='../outputs/param-search-tra-462-1000res-10folds-1e-06ridge-50w2vdim-<end>-13-09_21:18.csv'
    csv_file='../outputs/corpus462/start/reservoir_size-tra-462-1000res-10folds-1e-06ridge-50w2vdim-<start>-04-09_02:01.csv'
    plot_errors(csv_file_name=csv_file, plot_for=None, average_over='seed', rows=1,cols=1,
                x='reservoir_size', y='Meaning_Error', z='Sentence_Error')