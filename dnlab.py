#imports
import numpy as np
import matplotlib.pylab as plt
import pdb
import mdlab as mdl
import pandas as pd
import scipy
import scipy.spatial as ssp
import scipy.stats as sst
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split as split
import matplotlib.patches as mpatches

# Constants
n_neighbors = 15
rep_num = 250
bin_size = .01
training_prop=.5
resolution = .0001

frcc=mdl.FRCC()
trailing_type = frcc.trailing_type
trailing_param = frcc.get_trailing_param(trailing_type)
stats_test = frcc.stats_test

# defines the BinnedData object
class BinnedData(mdl.SpikeData):
    def __init__(self, header, spikes, bins, adaptation_rate, stim_adaptation_rate):
        mdl.SpikeData.__init__(self, header, spikes)
        self.bins=bins
        self.adaptation_rate = adaptation_rate
        self.stim_adaptation_rate = stim_adaptation_rate
    
    #searches for Binned_Data
    def get_kw_BinnedData(self, keyword, num):
        if keyword is 'electrode':
            newSpikeData = self.get_kw_SpikeData(electrode=num)
            bins = make_bins(newSpikeData)
        elif keyword is 'stim':
            newSpikeData = self.get_kw_SpikeData(stim=num)
            original_longest_stim = get_longest_stim(self)
            bins = make_bins(newSpikeData, longest_stim=original_longest_stim)
        else:
            print 'Invalid keyword'
            return
        ele_bins = BinnedData(newSpikeData.header, newSpikeData.spikes, bins, self.adaptation_rate, self.stim_adaptation_rate)
        return ele_bins
    
    # Takes a BinnedData object, splits it by electrode, and returns a list of BinnedData objects corresponding to each electrode
    def split_by_electrode(self, step_size=bin_size):
        split_data=[]
        for i in self.electrodes:
            split_data.append(self.get_kw_BinnedData(keyword='electrode', num=i))
        return split_data
    
    # Categorizes electrode as responding and nonresponding by median split. Returns a dict in format cat[electrode] = "fast"/"slow
    def categorize_by_ms(self):
        cat = {}
        #take stim ars by electrode
        ar = np.swapaxes(self.stim_adaptation_rate, 0, 1)[0]
        #find median of data
        median = np.median(ar)
        #loop through, and sort by high and low
        for index, ele_ar in enumerate(ar):
            if ele_ar > median:
                cat[index+1]="fast"
            else:
                cat[index+1]="slow"
        return cat
    

#converts a SpikeData object into a BinnedData object 
def Spike_to_Binned(SpikeData, step_size=bin_size, custom_stim_length = None):
    resolution = round(SpikeData.header.resolution.max(),4)
    bins=make_bins(SpikeData, step_size, custom_stim_length)
    adaptation_rate=get_adaptation_rate(SpikeData)
    stim_adaptation_rate = get_stim_adaptation_rate(SpikeData)
    newBinnedData=BinnedData(SpikeData.header, SpikeData.spikes, bins, adaptation_rate, stim_adaptation_rate)
    return newBinnedData 

#bins and normalizes spikes into windows
#can manually specify the duration of the longest stimulus or use the get_longest_stim function
def make_bins(SpikeData, step_size=bin_size, custom_stim_length = None):
    resolution = SpikeData.header.resolution.max()
    ap_matrix=[]
    spikes=SpikeData.spikes
    pre_stim_dur = SpikeData.header.pre_stim.max()
    if custom_stim_length is not None:
        longest_stim_dur = custom_stim_length
    else:
        longest_stim_dur = get_longest_stim(SpikeData)
    bin_start = int(pre_stim_dur/resolution)
    bin_end = int(((longest_stim_dur+.1)/resolution)+bin_start)
    for trial in spikes:
        trial = trial[bin_start:bin_end]
        nu=round(trial.shape[0]*resolution/step_size)
        split_trial=np.array(np.split(trial, nu))
        #adds spikes and normalizes to a z score
        ap_matrix.append(sst.zscore(np.sum(split_trial, axis=1)))
    bins=np.array(ap_matrix)
    # replaces all instances with no spikes into a 0
    #bins = np.nan_to_num(bins)
    return bins

#returns the longest stim from a SpikeData object
def get_longest_stim(SpikeData):
    stim_len = []
    for stim in SpikeData.stim:
        stim_len.append(SpikeData.extract_stim_dur(stim_code =stim))
    return round(np.max(stim_len),2)

# calculates average adaptation rates for each site from trials 5-25 in a SpikeData (or BinnedData)
def get_adaptation_rate(SpikeData):
    adaptation_rates={}
    ele_arr=np.unique(SpikeData.header.electrode)
    stim_arr = np.unique(SpikeData.stim)
    for i in ele_arr:
        ele_data=SpikeData.get_kw_SpikeData(electrode=i)
        ele_rates=[]
        for j in stim_arr:
            stim_data=ele_data.get_kw_SpikeData(stim=j)
            fr=stim_data.firing_rates(trailing_type, trailing_param, fr_type = 1)
            ar=mdl.adaptation_rate(fr)
            ele_rates.append(ar)
        adaptation_rates[i]=np.mean(ele_rates)
    return adaptation_rates

#calculates adaptation rate for a stimulus
def get_stim_adaptation_rate(SpikeData):
    adaptation_rates=[]
    ele_arr=SpikeData.electrodes
    ele_stim = np.unique(SpikeData.stim)
    for ele in ele_arr:
        ele_data=SpikeData.get_kw_SpikeData(electrode=ele)
        ele_rates=[]
        for stim in ele_stim:
            stim_data=ele_data.get_kw_SpikeData(stim=stim)
            fr=stim_data.firing_rates(trailing_type, trailing_param, fr_type = 2)
            ar=mdl.adaptation_rate(fr)
            ele_rates.append(ar)
        adaptation_rates.append(ele_rates)
    return np.array(adaptation_rates)


'''
Purpose: A K nearest neighbor machine learning algorithm used to calculate decoding accuracy. The script starts by
splitting the electrodes randomly by specifications determined by training proportion. After learning the training data, the alg uses KNN 
to guess which stim elicited a given neural response. Neuronal decoding accuracy is recorded


Input: SpikeData object corresponding to electrode, value of k, how many repetitions of slicing the data, what proportion of the data is used for training

Output: 2D array for decoding accuracy stim x trial number
'''
def decoding_accuracy(electrode, k=15, rep_num=250, training_prop=.5):
    # Generates parameters for acc and trial_usage
    num_stim = len(electrode.stim)
    total_trials = electrode.header.shape[0]
    trials_per_stim = total_trials/num_stim
    acc = np.zeros((num_stim, trials_per_stim))
    
    #2D array #2D arra`y, stim id x the trials per stimulus, increments up by one every time given trial used as target
    trial_usage = np.zeros((num_stim, trials_per_stim))
    
    indices = electrode.header.index.values
    stim_ids = np.array(electrode.header.stim)

    # knn algorithm
    for i in range(0,rep_num):
 
        #split data in half and predict values
        training_bins, target_bins, training_idx, target_idx, training_stim_ids, target_stim_ids = split(electrode.bins, indices, stim_ids, train_size=training_prop, test_size=1-training_prop)
        model = knn(n_neighbors=k, metric='euclidean')
        model.fit(training_bins, training_stim_ids)
        pred = model.predict(target_bins)
        

        #Checks to see if predictions are correct
        for j in range(0, pred.shape[0]):
            stim_id= target_stim_ids[j]
            idx = target_idx[j]
            stim_data = electrode.get_kw_SpikeData(stim=stim_id)
            net_trial_num = stim_data.header.loc[idx:idx].trial.max()
            mod_stim_data_header = stim_data.header.reset_index()
            true_trial_num = mod_stim_data_header[(mod_stim_data_header["trial"] == net_trial_num)].index[0]
            #increament trial usagebn
            trial_usage[stim_id-1][true_trial_num]+=1
            if pred[j] == stim_id:
                acc[stim_id-1][true_trial_num] += 1
            
    final_acc = np.divide(acc, trial_usage)
    return final_acc 

# takes npz data and splits it into a list
def separate_npz_data(npz_data):
    all_data = []
    bird_id_arr = np.unique(npz_data.birdid)
    for bird_id in bird_id_arr:
        spike_data =  npz_data.get_kw_SpikeData(birdid=bird_id)
        all_data.append(spike_data)
    return np.array(all_data)

def get_responding_xl(xl):
    responding_pairs = []
    responding_ids = np.sort(np.unique(xl.id))
    for bird_ele_id in responding_ids:
        loc = bird_ele_id.find("00")
        bird = bird_ele_id[loc+2:loc+3]
        ele = bird_ele_id[bird_ele_id.find("___")+3:]
        responding_pairs.append((bird, ele*100))
    return responding_pairs

#Multiplies the electrodes in a list of spike_data objects by n and returns the object
def convert_electrodes(split_data, n = 100):
    for spike_data in split_data:
        spike_data.electrodes = spike_data.electrodes * n
        spike_data.header.electrode = spike_data.header.electrode * n
    return

# does the knn algorithm on an array of experiments
def batch_decoding_accuracy(split_data, region = "N/A", n_neighbors=15, reps=250, training=.5):
    # Runs the decoding accuracy KNN algorithm on an array of spike data objects
    # split_data is the spike data array, bird_ele_class is a dataframe containing all responding electrodes and the 
    # region they are from.    
    # n neighbors is the amount of neighbors for the knn algorithm
    # reps is the amount of repitions of the algorithm to run
    # training is the training proportion
    
    c = ['birdid','electrode', 'region', 'hemisphere', 'average_adaptation_rate','average_decoding_accuracy','decoding_accuracy']
    final_df = pd.DataFrame(columns = c)
    #loops through split data
    n = 1
    for spike_data in split_data:
        ele_num_arr = spike_data.electrodes
        bird_tag = chr(int(str(spike_data.birdid[0])[0:2])) + chr(int(str(spike_data.birdid[0])[2:4]))
        bird_id = bird_tag + str(spike_data.birdid[0])[-5:-2]
        for x, ele_num in enumerate(ele_num_arr):
            ele_spike_data = spike_data.get_kw_SpikeData(electrode = ele_num)
            ele_binned_data = Spike_to_Binned(ele_spike_data)
            decoding_acc = decoding_accuracy(ele_binned_data, k=n_neighbors, rep_num=reps, training_prop=training)
            avg_decoding_acc = np.mean(decoding_acc)
            if ele_num < 1600:
                to_add = [bird_id, ele_num, region, "Left", get_adaptation_rate(ele_spike_data)[ele_num],avg_decoding_acc, decoding_acc]
                final_df = final_df.append(pd.DataFrame([to_add], columns=c)) 
            else:
                to_add = [bird_id, ele_num, region, "Right", get_adaptation_rate(ele_spike_data)[ele_num], avg_decoding_acc, decoding_acc]
                final_df = final_df.append(pd.DataFrame([to_add], columns=c))
        print str(n)+" Experiments Analyzed"
        n+=1
    return final_df.reset_index(drop=True)

# Checks an individual site/electrode to see if it responds to any stimuli
# To implement: options for statistical test
def site_check(site_data, frcc=frcc):
    trailing_type = frcc.trailing_type
    trailing_param = frcc.get_trailing_param(trailing_type)
    p_values = []
    for stim in site_data.stim:
        stim_data = site_data.get_kw_SpikeData(stim = stim)
        #baseline = stim_data.firing_rates(trailing_type, trailing_param, fr_type = 0)
        #stimulus = stim_data.firing_rates(trailing_type, trailing_param, fr_type = 1)
        #statistics, p_value = sst.wilcoxon(baseline, stimulus)
        fr = stim_data.firing_rates(trailing_type, trailing_param, fr_type = 2)
        base = np.zeros(len(fr))
        statistics, p_value = sst.ttest_rel(fr, base)
        #print baseline.shape
        p_values.append(p_value)
    return np.asarray(p_values)

# filters out nonresponding electrodes using site_check method
def res_filter(split_data, p_value= 0.01):
    # to prevent enormous amount of warnings
    import warnings; warnings.simplefilter('ignore')
    res_split_data = []
    for spike_data in split_data:
        res_ele = []
        for ele in spike_data.electrodes:
            ele_spike_data = spike_data.get_kw_SpikeData(electrode=ele)
            respond = site_check(ele_spike_data)
            r = respond < p_value
            if np.any(r) and not np.any(np.isnan((respond))):
                res_ele.append(ele)
        res_data = spike_data.get_kw_SpikeData(electrode = res_ele)
        res_split_data.append(res_data)
    return res_split_data