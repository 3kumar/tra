# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:17:45 2016

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


class TRA_Plotting(object):

    def __init__(self,csv_file_name=None):

        if csv_file_name is None:
            csv_file='outputs/tra-45-600res-10folds-50w2vdim-28-06_22:27.csv'
        else:
            csv_file=csv_file_name

        df= pd.read_csv(csv_file,sep=';')

        #read only specified columns and sort in order
        df=df[['Meaning_Error','Sentence_Error','spectral_radius','leak_rate','input_scaling','_instance']]
        #df=df.sort_values(['_instance','input_scaling','leak_rate','spectral_radius'])

        # get the unique value for all the parameters
        self.lr=np.unique(np.asarray(df["leak_rate"]))
        self.sr=np.unique(np.asarray(df["spectral_radius"]))
        self.iss=np.unique(np.asarray(df["input_scaling"]))

        #get the me,se for all parameter combination by averaging over different reservoir instances
        df=df.groupby(['input_scaling','spectral_radius','leak_rate']).mean()
        self.df_me=df['Meaning_Error'].reset_index()
        self.df_se=df['Sentence_Error'].reset_index()

    def plot_errors(self):

        # this will be used later to form grids of SR * LR for all input scaling
        param_space_dim=[len(self.iss),len(self.sr),len(self.lr)]

        #define ticks and lables for each axes
        xt=range(len(self.lr))
        xtl=list(self.lr)

        yt=range(len(self.sr))
        ytl=list(self.sr)

        #convert error to percentage and reshape with respect to parameter space
        #me=100*np.asarray(self.df_me["Meaning_Error"]).reshape(param_space_dim)

        me=100*np.asarray(self.df_se["Sentence_Error"]).reshape(param_space_dim)

        fig=plt.figure(9)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(3, 3),
                             direction="row",
                             axes_pad=0.5,
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
             ax.set_title('ISS: '+str(self.iss[i]))
             i+=1
             im = ax.imshow(me,cmap = 'seismic', interpolation='nearest', vmin=0, vmax=100,origin='lower', aspect='auto')
             ax.set_xticks(xt)
             ax.set_yticks(yt)

             ax.set_xticklabels(xtl)
             ax.set_yticklabels(ytl)
        fig.colorbar(im)
        plt.show()

if __name__=='__main__':
    tra_plot=TRA_Plotting()
    tra_plot.plot_errors()