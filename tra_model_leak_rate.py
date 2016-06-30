"""
Created on Sun Mar 27 13:40:22 2016

@author: fox
"""
#! /usr/bin/env python
import mdp
import csv
import itertools
import pickle
import time

from Oger.nodes import LeakyReservoirNode
from Oger.evaluation import n_fold_random,leave_one_out
from Oger.utils import rmse
from copy import deepcopy
from tra_error import ThematicRoleError

from reservoir_weights import generate_sparse_w, generate_sparse_w_in
from ridge_node import RidgeRegressionNode
#from Oger.nodes import RidgeRegressionNode
from tra_plot import PlotRoles

class ThematicRoleModel(ThematicRoleError,PlotRoles):

    def __init__(self,corpus=45,subset=range(15,41),reservoir_size= 600,input_dim=50,
                 spectral_radius=1.0, input_scaling= 0.75, bias_scaling=0,leak_rate=0.1,
                 ridge = 10**-6, _instance=0, n_folds = 0, seed=1, verbose=True):

                 self.COPRUS_W2V_MODEL_DICT='data/corpus-word-vectors-'+str(input_dim)+'dim.pkl'

                 #all the parameters required for ESN
                 self.input_dim=input_dim
                 self.reservoir_size = reservoir_size
                 self.spectral_radius = spectral_radius
                 self.input_scaling = input_scaling
                 self.seed=seed
                 self.bias_scaling=bias_scaling
                 self.subset=subset
                 self.ridge = ridge
                 self.leak_rate=leak_rate
                 self._instance=_instance
                 self.corpus=corpus
                 self.verbose=verbose

                 self.n_folds=n_folds
                 self.unique_labels=['N1-A1','N1-O1','N1-R1','N1-A2','N1-O2','N1-R2','N2-A1','N2-O1','N2-R1','N2-A2','N2-O2','N2-R2','N3-A1','N3-O1','N3-R1','N3-A2','N3-O2','N3-R2','N4-A1','N4-O1','N4-R1','N4-A2','N4-O2','N4-R2']

                 #load raw sentneces and labels from the files and compute several other meta info.
                 self.sentences,self.labels=self.load_corpus(corpus_size=corpus,subset=subset)
                 self.labels_to_index=dict([(label,index) for index,label in enumerate(self.unique_labels)])
                 self.sentences_len=[len(sent) for sent in self.sentences]
                 self.max_sent_len=max(self.sentences_len) #calculate max sentence length
                 self.sentences_offsets=[self.max_sent_len - sent_len for sent_len in self.sentences_len]
                 self.X_data,self.Y_data=self.__generate_input_data() # generate input data for ESN
                 self.initialize_esn()

    @staticmethod
    def load_corpus(corpus_size=45,subset=range(15,41)):
        if corpus_size==45:
           file_name='data/corpus_45.txt'
        if corpus_size==462:
           file_name='data/corpus_462.txt'

        sentences=[None]*len(subset)
        labels=[None]*len(subset)
        with open(file_name,'rb') as fh:
            i=0
            for index,line in enumerate(fh):
                if index in subset:
                   sentences[i],labels[i]=ThematicRoleModel.__process_line(line)
                   i+=1
        return sentences,labels

    @staticmethod
    def __process_line(line):
        s_line=line.strip().split('#')
        sentence=s_line[0].split('.')[1].strip()
        label=s_line[1].strip()
        sentence_words= sentence.split()
        sentence_labels= label.split()
        return sentence_words,sentence_labels

    def __generate_sentence_matrix(self,tokenized_sentence,sent_index):
        """
            tokenized_sentence:a list containing tokenized sentence where each element of list is a word
            i.e. ['dog','ate','the','bone']

            returns
                a matrix of dimension (sentence length * word2vec dimension)
        """
        sentence_matrix=mdp.numx.zeros((self.sentences_len[sent_index],self.input_dim))
        for idx,word in enumerate(tokenized_sentence):
            sentence_matrix[idx]=self.w2v_model[word.lower()]
        return sentence_matrix

    def __generate_label_matrix(self,tokenized_labels,sent_index,start='0'):
        """
            tokenized_labels: tokenized list of a labels index
                i.e. ['N1-A1','N2-O1','N1-A2','N2-O2'] corresponding labels indices in unique_labels
            index: index of sentence for which labels are given
            start: when to start giving teacher labels wrt to sentence, use 0 to start for the beginning
                    of sentence, 'end' to present at the end of sentence
            returns:
                a matrix of dimension (sentence length * len(self.unique_labels))
        """
        teaching_start= self.sentences_len[sent_index]-1 if start=='end' else 0

        # activate only the labels which are present in the sentence
        binary_label_array=mdp.numx.zeros((self.sentences_len[sent_index], len(self.unique_labels)))
        binary_label_array[:]=-1
        for lbl in tokenized_labels:
            binary_label_array[teaching_start:,self.unique_labels.index(lbl)]=1

        return binary_label_array

    def __generate_input_data(self):
        '''
            Generate the sequences for each sentences and labels to be used as input and output in ESN
            returns:
                x_data: create a list of sentence arrays where each array corresponds to a sentence and have shape (no of words,word2vec vector dimension)
                y_data: create a list of labels array for the corresponding sentence of dimensions (no of words, len of unique_labels)
        '''

        # Check if the w2v converted data format of raw sentence is available in the pkl file if yes then read from pickle file
        # else load word2vec model and generate a pickle file for further loading

        with open(self.COPRUS_W2V_MODEL_DICT,'r') as f:
            print 'Please Wait!! Loading data from file...'
            self.w2v_model=pickle.load(f) # data pickled as list where first element is sentences and second element is corresponding labels
            print 'Data Loaded Successfully.'

        x_data=[self.__generate_sentence_matrix(sentence,sent_index) for sent_index,sentence in enumerate(self.sentences)]
        y_data=[self.__generate_label_matrix(label,sent_index) for sent_index,label in enumerate(self.labels)]

        return (x_data, y_data)

    def initialize_esn(self,verbose=False):
        #generate sparse reservoir weights and input weights,
        w_r=generate_sparse_w(output_size=self.reservoir_size,specrad=self.spectral_radius,seed=self.seed)
        w_in=generate_sparse_w_in(output_size=self.reservoir_size,input_size=self.input_dim,scaling=self.input_scaling,seed=self.seed)
        w_bias=generate_sparse_w_in(output_size=1,input_size=self.reservoir_size,scaling=self.bias_scaling,seed=self.seed)

       ## Instansiate reservoir node, read-out and flow
        reservoir = LeakyReservoirNode(nonlin_func=mdp.numx.tanh,input_dim=self.input_dim,output_dim=self.reservoir_size,
                                            leak_rate=self.leak_rate,w=w_r,w_in=w_in,w_bias=w_bias,_instance=self._instance)

        read_out = RidgeRegressionNode(ridge_param=self.ridge, use_pinv=True, with_bias=True)
        #read_out = RidgeRegressionNode(ridge_param=self.ridge,other_error_measure= rmse,cross_validate_function=n_fold_random,n_folds=10,verbose=self.verbose)
        self.flow = mdp.Flow([reservoir, read_out],verbose=self.verbose)

    def trainModel(self,training_sentences,training_labels):
        '''
            inputs:
                training_sentences: Sentences on which ESN will be trained (list of arrays)
                training_labels: labels for corresponding training_sentences (list of arrays)
            returns:
                A copy of flow trained of the training_sentences
        '''
        f_copy=deepcopy(self.flow)# create a deep copy of initial flow for current train-test set
        data=[training_sentences,zip(training_sentences,training_labels)]
        f_copy.train(data)
        return f_copy

    def testModel(self,f_copy,test_sentences):
        '''
            f_copy: A copy of trained flow
            test_sentences= a list of senteces for testing

            return:
                list of arrays where each array is activation of the test sentence
        '''
        test_sentences_activations=[]
        for sent_index in range(len(test_sentences)):
            test_sentences_activations.append(f_copy(test_sentences[sent_index]))
        return test_sentences_activations

    def apply_nfold(self):
        #Split the data into training and test data depending on the n_folds
        #train_indices,test_indices are list of arrays containg indicies for training and testing correponding to folds
        if self.n_folds==0 or self.n_folds is None:
             train_indices=[range(len(self.sentences))] # train on all sentences
             test_indices=train_indices
        elif self.n_folds==1 or self.n_folds < 0: # if negative or 1
             train_indices, test_indices = leave_one_out(len(self.sentences))
        else:
             train_indices, test_indices = n_fold_random(n_samples=len(self.sentences),n_folds=self.n_folds)
        return train_indices, test_indices

    def execute(self,verbose=False):

        #instansiate the error and plot objects specified as parents class
        super(ThematicRoleModel,self).__init__()

        #obtain the training and test sentences by applying n_folds
        train_indices, test_indices=self.apply_nfold()

        # containers to receive mean rmse,meaning and sentence error for test sentences on all the folds
        all_mean_meaning_err = []
        all_mean_sentence_err = []
        all_mean_rmse = []

        iteration = range(len(train_indices))
        #prepare a list of training and test sentences arrays based on train_indices and test_indices respectively
        for fold in iteration:

            print 'Fold- %d'%(fold+1)
            #generating training sentences and labels data for each fold
            curr_train_sentences=[self.X_data[index] for index in train_indices[fold]]
            curr_train_labels=[self.Y_data[index] for index in train_indices[fold]]

            #generating test sentences and labels data for each fold
            curr_test_sentences=[self.X_data[index] for index in test_indices[fold]]
            curr_test_labels=[self.Y_data[index] for index in test_indices[fold]]

            test_sentences_subset=[index for index in test_indices[fold]]

            # Training:- return a flow trained on current fold training sentences
            f_copy=self.trainModel(curr_train_sentences,curr_train_labels)

            fold_meaning_error=[]
            fold_sentence_error=[]
            fold_rmse=[]

            #Testing:- collect activations of all test sentences in current fold in a list
            test_sentences_activations=self.testModel(f_copy,curr_test_sentences)
            for sent_no,sent_activation in enumerate(test_sentences_activations):
                    #compute error method returns a tuple of errors
                    errors=self.compute_error(sent_activation,curr_test_labels[sent_no],fold)
                    meaning_error, sentence_error=errors
                    fold_meaning_error.append(meaning_error)
                    fold_sentence_error.append(sentence_error)
                    fold_rmse.append(rmse(sent_activation,curr_test_labels[sent_no]))

            all_mean_rmse.append(mdp.numx.mean(fold_rmse))
            all_mean_meaning_err.append(mdp.numx.mean(fold_meaning_error))
            all_mean_sentence_err.append(mdp.numx.mean(fold_sentence_error))

            #self.plot_outputs(test_sentences_activations,test_sentences_subset,plot_subtitle='')

        if verbose:
            print '\n mean rmse::',mdp.numx.mean(all_mean_rmse)
            print 'SD in mean nrmse::',mdp.numx.std(all_mean_rmse)
            print '\n mean meaning error::',mdp.numx.mean(all_mean_meaning_err)
            print 'SD in mean meaning error::',mdp.numx.std(all_mean_meaning_err)
            print '\n mean sentence error::',mdp.numx.mean(all_mean_sentence_err)
            print 'SD in mean sentence error::',mdp.numx.std(all_mean_sentence_err)

        return mdp.numx.mean(all_mean_rmse),mdp.numx.std(all_mean_rmse),\
                mdp.numx.mean(all_mean_meaning_err),mdp.numx.std(all_mean_meaning_err), \
                mdp.numx.mean(all_mean_sentence_err),mdp.numx.std(all_mean_sentence_err)

    def grid_search(self,output_csv_name=None,progress=True,verbose=False):
        '''
            this execute method does a grid search over reservoir parameters and log the errors in a csv file w.r.t to
            gridsearch parameters

        '''
        ct=time.strftime("%d-%m_%H:%M")
        if output_csv_name is None:
            out_csv='outputs/tra-'+str(self.corpus)+'-'+\
                     str(self.reservoir_size)+'res-'+\
                     str(self.n_folds)+'folds-'+\
                     str(self.input_dim)+'w2vdim-'+\
                     ct+'.csv'
        else:
            out_csv=output_csv_name

        #dictionary of parameter to do grid search on
        #Note the parameter key should match the name with variable of this class
        gridsearch_parameters = {
                                'spectral_radius':mdp.numx.arange(0.8, 1.5, 0.1),
                                'input_scaling':mdp.numx.arange(0.5, 1.5, 0.15),
                                'leak_rate':mdp.numx.arange(0.1,0.5,0.05)
                                }
        parameter_ranges = []
        parameters_lst = []

        # Construct the parameter space
        # Loop over all nodes that need their parameters set
        for node_key in gridsearch_parameters.keys():
            # Loop over all parameters that need to be set for that node
            # Append the parameter name and ranges to the corresponding lists
                parameter_ranges.append(gridsearch_parameters[node_key])
                parameters_lst.append(node_key)

        # Construct all combinations
        param_space = list(itertools.product(*parameter_ranges))
        if progress:
            iteration = mdp.utils.progressinfo(enumerate(param_space), style='timer', length=len(param_space))
        else:
            iteration = enumerate(param_space)

        # Loop over all points in the parameter space i.e for each parameters combination
        with open(out_csv,'wb+') as csv_file:
            w=csv.writer(csv_file,delimiter=';')
            csv_header=['S.No','RMSE','std. RMSE','Meaning_Error','std. Meaning Error', 'Sentence_Error','std. Meaning Error']
            csv_header+=[param for param in parameters_lst]
            w.writerow(csv_header)

            for paramspace_index_flat, parameter_values in iteration:
                # Set all parameters of all nodes to the correct values
                for parameter_index, parameter in enumerate(parameters_lst):
                    # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized
                    self.__setattr__(parameter,parameter_values[parameter_index])

                # Re-initialize esn
                self.initialize_esn()

                # Do the validation and get the errors for each paramater combination
                errors = self.execute()

                # Store the current errors in the respective errors arrays for a param combination
                mean_rmse=errors[0]
                std_rmse=errors=[1]
                mean_meaning_error =  errors[2]
                std_meaning_error =  errors[3]
                mean_sentence_error =  errors[4]
                std_sentence_error =  errors[5]

                row=[paramspace_index_flat+1,mean_rmse,std_rmse, mean_meaning_error, std_meaning_error,mean_sentence_error,std_sentence_error]
                row+=list(param_space[paramspace_index_flat])
                w.writerow(row)

if __name__=="__main__":
    corpus=462
    subset=range(0,462)
    model_instances=1
    ridge=mdp.numx.power(10, mdp.numx.arange(-10,0,1))
    model = ThematicRoleModel(corpus=corpus,input_dim=50,reservoir_size=1200,input_scaling=0.75,spectral_radius=1.0,
                            leak_rate=0.15,bias_scaling=0,ridge=1e-6,subset=subset,n_folds=10,verbose=True)

    inst_meaning_error=[]
    inst_sent_error=[]
    for instance in range(model_instances):
        rmse,std_rmse,meaning_error,std_me,sentence_error,std_se=model.execute(verbose=True)
        inst_meaning_error.append(meaning_error)
        inst_sent_error.append(sentence_error)
    print 'errors: ', (mdp.numx.mean(inst_meaning_error),mdp.numx.mean(inst_sent_error))

    #model.grid_search()