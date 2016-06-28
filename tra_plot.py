import pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot

class PlotRoles(object):

    def __init__(self,save_pdf=True,nr_nouns=4, nr_verbs=2,file_name="plots/testing1",
                window=0, verbose=False, y_lim=[-1.5,1.5], no_ext_fct=True):

            super(PlotRoles,self).__init__()

            self.save_pdf=save_pdf
            self.nr_nouns=nr_nouns
            self.nr_verbs=nr_verbs
            self.file_name=file_name
            self.window=window
            self.verbose=verbose
            self.y_lim=y_lim
            self.no_ext_fct=no_ext_fct


    def get_labels(self,test_sentences_subset,verbose=False):
        """
        Data is organised in this way:
            - l_data is a list of sentences
            - each sentence is a list of words
        The label ticks are generated. Keep in mind that when plotting the ticks are plot like if there is an initial pause (because the tick doesn't begin at x=0).
        - offset: represents the difference between the maximum number of words in the data and the number of word of a given sentence.
        - l_offset: is the list of offset for each sentence of l_data
        """
        tokenized_sentence = [self.sentences[i] for i in test_sentences_subset]
        sent_ticks = len(tokenized_sentence)*[None]

        for sent_index in range(len(tokenized_sentence)):
            sent_ticks[sent_index]=[] # ticks list for each sentence
            for word_index in range(len(tokenized_sentence[sent_index])):
                sent_ticks[sent_index].append(tokenized_sentence[sent_index][word_index])
        return (tokenized_sentence, sent_ticks)

    def plot_outputs(self,outputs,test_sentences_subset,plot_subtitle):

        print " *** Plotting outputs *** "

        (labels, lab_tick) = self.get_labels(test_sentences_subset)
        TOSpN = 3*self.nr_verbs  #number of Total Output Signal per Noun (for AOR, it's 3 times the number of verbs)

        ## Plotting logic
        if self.save_pdf:
            ## Initiate object PdfPages for saving figures
            pp = PdfPages(str(self.file_name)+'_'+str(plot_subtitle)+'.pdf')
        for i in range(len(outputs)):
            ## For each sentence, plot as many graphs  as the number of nouns
            for j in range(self.nr_nouns):
                pl.figure()
                pl.plot(outputs[i][:,TOSpN*j:TOSpN*(j+1)])
                pl.legend(self.unique_labels[TOSpN*j:TOSpN*(j+1)], loc='upper left')
                pl.suptitle("Testing sentence "+str(self.subset[i])+ ": '"+" ".join(labels[i])+"'"+"\n"+plot_subtitle)
                pl.xticks(range(0,outputs[i].shape[0]),lab_tick[i])
                pl.margins(0.15)
                a = matplotlib.pyplot.gca()
                if self.y_lim!=None:
                    a.set_ylim(self.y_lim)
                if self.save_pdf:
                    # Save figure for each plot
                    pp.savefig()
                pl.close()

        if self.save_pdf:
            ## Close object PdfPages
            pp.close()
        print "*** Plotting finished ***"



    def plot_with_output_fashion(self,l_array, subset, d_io, root_file_name, subtitle="_output_fashion", legend=None, y_lim=None, verbose=False):
        print " *** Plotting with output fashion *** "
        print " * root_file_name="+root_file_name+" - "+subtitle+" * "

        (labels, lab_tick) = self.get_labels(l_data=d_io['l_data'], subset=subset,
                                        initial_pause=d_io['initial_pause'], l_offset=d_io['l_offset'])

        pp = PdfPages(str(root_file_name)+'_'+str(subtitle)+'.pdf')

        for i in range(len(l_array)):
            if verbose:
                print "idx_sentence", subset[i]
                print "i="+str(i)
                print "output[i]", l_array[i]
                print "label_sentence", labels[i]
                print "len(label_sentence)", len(labels[i])
                print "words_tick", lab_tick[i]
            pl.figure()
            pl.plot(l_array[i])
            if legend is not None:
                pl.legend(legend)

            pl.suptitle("Testing sentence "+str(subset[i])+ ": '"+" ".join(labels[i])+"'"+"\n"+subtitle)
            pl.xticks(range(d_io['act_time'],l_array[i].shape[0],d_io['act_time']))
            a = matplotlib.pyplot.gca()
            if y_lim!=None:
                a.set_ylim(y_lim)
            a.set_xticklabels(lab_tick[i], fontdict=None, minor=False)

            pp.savefig()
            pl.close()
        pp.close()
        print " * Plot finished * "
        print " *** "

    def plot_array_in_file(root_file_name, array_, data_subset=None, titles_subset=None, plot_slice=None, title="", subtitle="", legend_=None):
        """
        inputs:
            array_: is the array or matrix to plot
            data_subset: correspond to the subset of the whole data that is treated. array_ is corresponds to this subset. /
                array_ and subset have to have the same length
            titles_subset: list of subtitles
            plot_slice: slice determining the element of array_ that will be plotted.
        """
        import mdp
        if data_subset is None:
            data_subset = range(len(array_))
        if titles_subset is None:
            titles_subset = ['' for _ in range(len(data_subset))]
            nl_titles_sub = ''
        else:
            nl_titles_sub = '\n'
        if array_==[] or array_==mdp.numx.array([]):
            import warnings
            warnings.warn("Warning: array empty. Could not be plotted. Title:"+str(title))
            return
        if plot_slice is None:
            plot_slice = slice(0,len(data_subset))
        else:
            if (plot_slice.stop-1) > len(data_subset):
                raise Exception, "The last element of the slice is out of the subset."
            subtitle = subtitle+"_slice-"+str(plot_slice.start)+"-"+str(plot_slice.stop)+"-"+str(plot_slice.step)
        ppIS = PdfPages(str(root_file_name)+str(title)+'.pdf')

        for i in range(plot_slice.stop)[plot_slice]:
            pl.figure()
            pl.suptitle(title+" "+str(titles_subset[i])+nl_titles_sub+" - seq "+str(data_subset[i])+"\n"+subtitle)
            pl.plot(array_[i])
            if legend_ is not None:
                pl.legend(legend_)
            ppIS.savefig()
            pl.close()
        ppIS.close()

