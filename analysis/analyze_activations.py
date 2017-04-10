# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:21:23 2016

This script is used to create subplots for activations of given sentences.

@author: fox
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

try:
    import cPickle as pickle
except:
    import pickle as pickle

def load_activations(corpus='45'):
    """
        returns:
            sentence_activations: list of arrays where each array is the activation of a sentence
            tokenized_sentences: list of list of tokenizes sentences
    """
    data_file='../outputs/activations/corpus-'+corpus+'.act'
    with open(data_file,'r') as f:
        data=pickle.load(f)
        sentence_activations=data[0]
        tokenized_sentences=data[1]
    return sentence_activations,tokenized_sentences

def plot_outputs(corpus='45',subplots=2,plot_noun=2,sentences_order=None,noun_tick_pos=None,verbs_tick_pos=None):
    """
        subplots: No of subplots to be drawn
        sentence_order:list of sentences numbers in order to be drawn in subplots

    Note: no of subplots should be same as length of sentences_order
    """

    if subplots!=len(sentences_order):
        raise Exception("No of subplots should be same length of sentence order")
    elif noun_tick_pos is None or verbs_tick_pos is None:
        print Warning("List of tick positions for verbs and noun-%d in the sentences is missing. Noun-%d in the ticks will not be marked"%(plot_noun,plot_noun))

    sent_activations,tok_sentences=load_activations(corpus=corpus)

    plt.close('all')

    if subplots==2:
        f, axs= plt.subplots(1, 2, sharey='row')
    elif subplots==4:
        f, axs = plt.subplots(2, 2, sharey='row')

    for ax_index,ax in enumerate(f.axes):
        sent_no=sentences_order[ax_index]
        s_act=sent_activations[sent_no][0:-1,:]
        tok_sent=tok_sentences[sent_no]

        for j in range(nr_nouns):
            if j==plot_noun-1:
                ax.plot(s_act[:,TOSpN*j:TOSpN*(j+1)])
                if ax_index==0:
                    ax.legend(unique_labels[TOSpN*j:TOSpN*(j+1)],fancybox=True,loc="upper left").get_frame().set_alpha(0.4)

                ax.set_title("("+str(ax_index+1)+ ") "+" ".join(tok_sent[1:-1])+"", loc='left')
                ax.set_xticks(range(0,s_act.shape[0]+1))
                tok_sent[0]=""
                tok_sent[-1]=""
                ax.set_xticklabels(tok_sent,rotation=60)

                if noun_tick_pos is not None or verbs_tick_pos is not None:
                    noun_color_tick=noun_tick_pos[ax_index]
                    for vn,verb_color_tick in enumerate(verbs_tick_pos[ax_index]):
                        ax.get_xticklabels()[verb_color_tick].set_color('green')
                    ax.get_xticklabels()[noun_color_tick].set_color('red')

                ax.axhline(y=0, c="brown",ls='--', linewidth=1)
                ax.grid(alpha=0.6,c='grey',ls='dotted')
                ax.set_ylim([-2.0,2.0])

    f.subplots_adjust(left=0.05,right=0.95, top=0.95,bottom=0.1,hspace=0.25, wspace=0.1)
    plt.show()
    f.set_size_inches(12, 10)
    plt.savefig('/home/fox/thesis_report/src/act_analysis_1.pdf')

if __name__=="__main__":

    pgf_with_latex = {
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif":[],
        'font.size': 14,                   # blank entries should cause plots to inherit fonts from the document
        'axes.titlesize':'medium',
        "axes.labelsize": 'medium',
        "legend.fontsize": 12,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        }

    plt.rcParams.update(pgf_with_latex)

    nr_nouns=4
    TOSpN = 3*2  #number of Total Output Signal per Noun (for AOR, it's 3 times the number of verbs)
    unique_labels=['N1-A-V1','N1-O-V1','N1-R-V1','N1-A-V2','N1-O-V2','N1-R-V2',
                   'N2-A-V1','N2-O-V1','N2-R-V1','N2-A-V2','N2-O-V2','N2-R-V2',
                   'N3-A-V1','N3-O-V1','N3-R-V1','N3-A-V2','N3-O-V2','N3-R-V2',
                   'N4-A-V1','N4-O-V1','N4-R-V1','N4-A-V2','N4-O-V2','N4-R-V2']


    sentences_order=[462+17,462+23,462+27,462+16]
    noun_tick_pos=[5,5,5,7]
    verbs_tick_pos=[[3],[3,7],[3,8],[4]]

    '''sentences_order=[462+0,462+1,462+10,462+11]
    noun_tick_pos=[1,2,1,2]
    verbs_tick_pos=[[2,8],[4,10],[2,8],[4,10]]'''

    plot_outputs(corpus='462_45',subplots=4,plot_noun=2,sentences_order=sentences_order,noun_tick_pos=noun_tick_pos,verbs_tick_pos=verbs_tick_pos)