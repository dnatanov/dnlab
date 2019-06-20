# -*- coding: utf-8 -*-
"""
Mingwen Dong Lab functions: mdlab

@author: md
"""

# import required modules
import numpy as np
import os, glob
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import scipy
import scipy.stats
import scipy.io
from scipy import signal
import statsmodels.api as sm
from sklearn import linear_model
import pdb
import gc

# common parameters
data_folder = "./data/"  # folder where data is stored
figure_folder = "./figure/"
separator = "___"  # separator to combine string
resolution = 5e-4
mua_sua_distinguisher = 100  # if an electrode number can be divieded by 100 exactly, it is MUA. otherwise, SUA.
robust_adr_propout = 0.2  # proption of responses to leave out when calculating robust adapation rate

# recommended constants and archives codes
# paper_figsize = (4, 2.5)
# poster_figsize = (8, 6)
# data analysis constant
# DAC = {
#     "prop_cut": 0.0, # proportion of the data to trim from both sides to caculate the robust mean
# }


class Folder(object):
    """
    store the default folder to store figure and results
    """
    def __init__(self, data_type):
        self.__figure = "figure"
        self.__result = "result"
        self.__mua = "mua"
        self.__sua = "sua"
        self.__data = "data"
        self.__data_type = data_type

    def figure(self):
        directory = "./%s_%s/" % (self.__data_type, self.__figure)
        if self.__data_type == self.__mua:
            return directory
        elif self.__data_type == self.__sua:
            return directory
        else:
            raise ValueError("Unknown data type: only support 'sua' & 'mua'. ")

    def result(self):
        directory = "./%s_%s/" % (self.__data_type, self.__result)
        if self.__data_type == self.__mua:
            return directory
        elif self.__data_type == self.__sua:
            return directory
        else:
            raise ValueError("Unknown data type: only support 'sua' & 'mua'. ")

    def data(self):
        directory = "./%s/" % self.__data
        return directory


class PARAM(object):
    resolution = resolution
    mua_sua_distinguisher = mua_sua_distinguisher
    outlier_threshold = 3.5
    stim_waveform_start_idx = 4 # stimulus waveform starts from 4th column in stim_waveforms

    # parameters used to test whether an electrode is responsive or not
    trial_per_test = 40  # neural responses are checked every these trials to ensure stationary responses
    stats_test = "binomial"  # what statistic test are used
    fr_lb = 1e-2  # stim - baseline should be positive
    sig_level = 0.05


class FRCC(object):
    # firing rate calculation constant
    def __init__(self,
                 trailing_type="proportional",  #
                 trailing_prop=0.1,  #
                 trailing_duration=0.1,  #
                 bs_prop_out=0.0,  #
                 bs_outlier_factor=3.5,  #
                 stats_test="binomial",  # test used to responsiveness measurement
                 bs_smooth_type="robust_avg",
                 bs_smooth_mg=10):
        """

        :param trailing_type: "fixed_duration" or "proportional"
        :param trailing_prop: trailing duration after stimulus offset (10% of stimulus duration)
        :param trailing_duration: number of seconds after stimulus offset
        :param bs_prop_cut: proportion to leave out when calculate a robust mean
        :param bs_outlier_factor: number of standard deviations to be considered as an outlier baseline
        :param bs_smooth_type: smoothing method used during baseline calculation
            "raw": baseline firing rate without smoothing
            "avg": mean
            "robust_avg":
            "linreg":
            "robust_linreg":
            "local_smooth": # for local_smooth baseline at trial iii = scipy.stats.trim_mean(bs[iii-bs_smooth_mg : iii+bs_smooth_mg])
        :param bs_smooth_mg:
        """
        self.trailing_type = trailing_type
        self.trailing_prop = trailing_prop
        self.trailing_duration = trailing_duration
        self.bs_prop_out = bs_prop_out
        self.bs_outlier_factor = bs_outlier_factor
        self.bs_smooth_type = bs_smooth_type
        self.bs_smooth_mg = bs_smooth_mg
        self.stats_test = stats_test

    def get_trailing_param(self, trailing_type=None):
        """
        return corresponding trailing parameter based on trailing type
        :param trailing_type: "fixed_duration" or "proportional"
        :return:
        """
        if trailing_type is None:
            trailing_type = self.trailing_type
        if trailing_type == "fixed_duration":
            return (self.trailing_duration)
        elif trailing_type == "proportional":
            return (self.trailing_prop)
        else:
            print("Wrong trailing type in frcc!")

    def add_trailing_time(self, dur):
        if self.trailing_type == "fixed_duration":
            return (dur + self.get_trailing_param())
        if self.trailing_type == "proportional":
            return (dur * (1 + self.get_trailing_param()))
        else:
            raise ValueError("Wrong trailing type in frcc!")


frcc = FRCC()


def response_criterion(res_type):
    class MUA_RC(object):
        pval = 0.05
        lb = 10
        ub = 10000

    class SUA_RC(object):
        pval = 0.05
        lb = 0.1
        ub = 10000

    res_dict = {"sua": SUA_RC, "mua": MUA_RC}
    return (res_dict[res_type])


class SpikeData(object):
    """
    create a spike data object based on the header and spike matrix for easy data manipulation.
    """

    def __init__(self, header, spikes):
        """
        initialize the data
        :param header: header_df
        :param spikes: spike matrix
        """
        self.header = header
        self.spikes = spikes
        self.electrodes = np.unique(header["electrode"])
        self.birdid = np.unique(header["birdid"])
        self.stim = np.unique(header["stim"])
        self.resolution = header["resolution"].values[0]
        self.trial = np.unique(header["trial"])
        self.pre_stim = header.pre_stim.values[0]
        self.header_keywords = np.asarray(header.columns)
        
        # DIN modification
        #self.stims = stims
        #if "id" not in self.header_keywords:
            #self._id2header()

    def get_index(self, shift=0, **keyword_params):
        """
        return the index and logical index that satisfy the conjunction of keywords.\n
        possible kerwords are the keys in the header_df.\n
        Example keyword: electrode, stim, birdid, trial
        shift: get the indexes with corresponding shift, \n
            e.g., -1 means get the trial before the kw specified
        """
        logidx = np.ones(len(self.header), dtype=bool)
        for condition in keyword_params:
            condition_values = keyword_params[condition]
            condition_logidx = np.zeros(len(self.header), dtype=bool)
            if hasattr(condition_values, "__iter__"):  # check whether the values is iterable
                for iii, condition_value in enumerate(condition_values):
                    tmp_logidx = self.header[condition] == condition_value
                    condition_logidx += tmp_logidx
            else:
                condition_logidx = self.header[condition] == condition_values
            logidx = np.logical_and(logidx, condition_logidx)
        absidx = np.flatnonzero(logidx) + shift
        logidx = np.zeros(logidx.shape, dtype=bool)
        logidx[absidx] = True
        return absidx, logidx

    def _get_logidx_header(self, logidx):
        return self.header[logidx]

    def _get_idx_spikes(self, absidx):
        if np.all(np.diff(absidx) == 1):  # if indexes are all connected, use fancy index to speed up
            return (np.copy(self.spikes[absidx[0]:absidx[-1] + 1, :]))
        return self.spikes[absidx, :]

    def get_idx_SpikeData(self, absidx, logidx):
        """
        return the sub-object that contains the corresponding
        """
        return SpikeData(self._get_logidx_header(logidx), self._get_idx_spikes(absidx))

    def get_kw_SpikeData(self, shift=0, **keyword_params):
        """
        return the SpikeData that consists of the subset of the data that satisfy\n
        the conditions listed in the keywords.\n
        any property of the header can be used for subsetting.\n
        Common example: electrode, stim, trial, birdid.\n
        shift=0: return data satisfies the condition,
            shift=-1: return data whose index = index - 1

        """
        logidx = np.ones(len(self.header), dtype=np.bool)
        for condition in keyword_params:
            condition_values = keyword_params[condition]
            condition_logidx = np.zeros(len(self.header), dtype=bool)
            if hasattr(condition_values, "__iter__"):  # check whether the values is iterable
                for iii, condition_value in enumerate(condition_values):
                    tmp_logidx = self.header[condition] == condition_value
                    condition_logidx += tmp_logidx
            else:
                condition_logidx = self.header[condition] == condition_values
            logidx = np.logical_and(logidx, condition_logidx)
        absidx = np.flatnonzero(logidx) + shift
        logidx = np.zeros(logidx.shape, dtype=bool)
        logidx[absidx] = True
        return SpikeData(self._get_logidx_header(logidx), self._get_idx_spikes(absidx))

    def extract_stim_dur(self, stim_code):
        """
        !!!use with caution!!! Make sure the stimulus code is umambiguous.\n
        birdid and other specific conditions are required to give right value.\n
        get the stimulus duration for the coded stimulus.\n
        """
        ex_electrode = self.electrodes[0]
        ex_data = self.get_kw_SpikeData(electrode=ex_electrode)
        tmp_logidx = ex_data.header.stim == stim_code
        tmp_header = ex_data.header[tmp_logidx]
        stim_dur = np.mean(tmp_header.ending_time - tmp_header.starting_time)
        return (stim_dur)

    def electrode_code_adjustment(self, adjustment=0):
        """
        adjust the electrode code if it's coded in a wrong way
        :param adjustment: number to subtract by, e.g., adjustment=1 will make the electrode smllaer by 1
        :return:
        """
        self.header.electrode -= adjustment
        self.electrodes = np.unique(self.header.electrode)

    def mua_or_sua(self, type, param=PARAM):
        """
        extract MUA or SUA
        :param type: "mua" OR "sua"
        :param param: constant defined in mdlab
        :return:
        """
        if type == "mua":
            logidx = self.header.electrode % param.mua_sua_distinguisher == 0
            absidx = np.flatnonzero(logidx)
            return (self.get_idx_SpikeData(absidx, logidx))
        elif type == "sua":
            logidx = self.header.electrode % param.mua_sua_distinguisher != 0
            absidx = np.flatnonzero(logidx)
            return (self.get_idx_SpikeData(absidx, logidx))
        else:
            raise ValueError("Wrong data type!")

    def compress_data(self, step_size):
        """
        reduce the resolution to save up space.\n
        :param step_size: number of bins to combined (step sum)
        :return: flag indicates whether compression is successful or not.
        """
        try:
            self.resolution *= step_size
            self.spikes = step_sum(self.spikes, step_size=step_size, axis=1)
            self.header["resolution"] *= step_size
            return 1
        except:
            raise ValueError("Compress failed!")

    def save_data(self, filename):
        """
        save the current spike data as a .npz file.
        :return:
        """
        save_npz_data(filename, self.header, self.spikes)

    def bird_firing_rates(self, birdid=None, fr_type=2, frcc=frcc):
        """
        calculate the firing rate for all electrodes and all trials from one bird
        :param birdid: firing rate to get
        :param fr_type: 0, bs_fr, 1: stim_fr, 2: net_fr, 3: all of them
        :param frcc: firing rate calculation constant
        :return:
          frs_list: a 3d numpy array, [net_frs, stim_frs, bs_frs] * # of trials * # of electrodes
          ele_id_list:
        """
        trailing_type = frcc.trailing_type
        trailing_param = frcc.get_trailing_param(trailing_type)

        if birdid is not None:
            bird_data = self.get_kw_SpikeData(birdid=birdid)
        else:
            bird_data = self

        ele_id_list = []
        net_frs_list = []
        bs_frs_list = []
        stim_frs_list = []
        for ele_id in bird_data.electrodes:
            ele_id_list.append(ele_id)
            ele_data = bird_data.get_kw_SpikeData(electrode=ele_id)
            bs_frs, stim_frs, _ = ele_data.firing_rates(trailing_type, trailing_param, fr_type=3)
            smooth_bs_frs = smooth_baseline(bs_frs, frcc=frcc)
            net_frs = stim_frs - smooth_bs_frs
            net_frs_list.append(net_frs)
            bs_frs_list.append(bs_frs)
            stim_frs_list.append(stim_frs)
        frs_list = np.asarray([bs_frs_list, stim_frs_list, net_frs_list])
        frs_list = np.swapaxes(frs_list, 1, 2)
        if fr_type <= 2:
            return (frs_list[fr_type], ele_id_list)
        else:
            return (frs_list, ele_id_list)

    def net_firing_rates(self, frcc=frcc):
        """
        calculate the firing rate for all trials in one electrode without separating for different stimuli.
        :param fr_type: 0, bs_fr, 1: stim_fr, 2: net_fr, 3: all of them
        :param frcc: firing rate calculation constant
        :return:
          frs_list: a 3d numpy array, [net_frs, stim_frs, bs_frs] * # of trials
        """
        trailing_type = frcc.trailing_type
        trailing_param = frcc.get_trailing_param(trailing_type)
        bs_frs, stim_frs, _ = self.firing_rates(trailing_type, trailing_param, fr_type=3)
        smooth_bs_frs = smooth_baseline(bs_frs, frcc=frcc)
        net_frs = stim_frs - smooth_bs_frs
        return (net_frs)

    def ele_firing_rates(self, fr_type=2, frcc=frcc):
        """
        To complete
        calculate the firing rate for all trials in one electrode
        :param fr_type: 0, bs_fr, 1: stim_fr, 2: net_fr, 3: all of them
        :param frcc: firing rate calculation constant
        :return:
          frs_list: a 3d numpy array, [net_frs, stim_frs, bs_frs] * # of trials
        """
        trailing_type = frcc.trailing_type
        trailing_param = frcc.get_trailing_param(trailing_type)
        bs_frs, stim_frs, _ = self.firing_rates(trailing_type, trailing_param, fr_type=3)
        smooth_bs_frs = smooth_baseline(bs_frs, frcc=frcc)
        net_frs = stim_frs - smooth_bs_frs

        stim_frs_list = []
        net_frs_list = []
        bs_frs_list = []
        for stim in np.sort(self.stim):
            _, logidx = self.get_index(stim=stim)
            net_frs_list.append(net_frs[logidx])
            bs_frs_list.append(bs_frs[logidx])
            stim_frs_list.append(stim_frs[logidx])
        frs_list = [bs_frs_list, stim_frs_list, net_frs_list]
        if fr_type <= 2:
            return (frs_list[fr_type])
        else:
            return (frs_list)

    def idx_firing_rates(self, absidx, frcc=None):
        """
        get the firing rate from specific trials (only applicable to spikedata to individual unit/electrode)
        :param absidx: index of required firing rates
        :param trailing_type: default "proportional"
        :param trailing_param: default 0.1
        :param smooth_type: baseline smoooth method, default "robust_linreg" (robust linear regression)
        :param smooth_mg: smooth magnitude used for local smooth, only applicable for local_smooth method
        :return:
            raw_bs_fr, stim_fr, net_fr (stim - smooth bs)
        """
        trailing_type = frcc.trailing_type
        trailing_prop = frcc.trailing_prop
        trailing_duration = frcc.trailing_duration

        trailing_param = frcc.get_trailing_param(trailing_type)

        bs_frs = self.firing_rates(trailing_type=trailing_type, trailing_param=trailing_param, fr_type=0)

        tmp_header = self.header.iloc[0]
        st_time = tmp_header.pre_stim
        st_index = int(st_time / tmp_header.resolution)
        num_items = len(absidx)
        stimulus_frs = np.zeros(num_items)
        for iii, idx in enumerate(absidx):
            tmp_header = self.header.iloc[idx]
            tmp_spiketrain = self.spikes[idx, :]
            stim_dur = tmp_header.ending_time - tmp_header.starting_time
            trailing_dur = 0.0
            if trailing_type == "proportional":
                trailing_dur = stim_dur * trailing_prop
            elif trailing_type == "fixed_duration":
                trailing_dur = trailing_duration
            else:
                print("Wrong trailing type!")
            ed_time = st_time + stim_dur + trailing_dur
            ed_index = int(ed_time / tmp_header.resolution)
            stimulus_frs[iii] = np.sum(tmp_spiketrain[st_index:ed_index]) / (stim_dur + trailing_dur)
        smoothed_bs_frs = smooth_baseline(bs_frs, frcc)
        idx_smooth_bs_frs = smoothed_bs_frs[absidx]
        idx_bs_frs = bs_frs[absidx]
        net_frs = stimulus_frs - idx_smooth_bs_frs

        return ([idx_bs_frs, stimulus_frs, net_frs])

    def stim_firing_rates(self, stim, frcc=None):
        """
        get the firing rate for a stimulus (only applicable to spikedata to individual unit/electrode)
        :param stim: firing rate for the stimulus one want
        :param trailing_type: default "proportional"
        :param trailing_param: default 0.1
        :param smooth_type: baseline smoooth method, default "robust_linreg" (robust linear regression)
        :param smooth_mg: smooth magnitude used for local smooth, only applicable for local_smooth method
        :return:
            raw_bs_fr, stim_fr, net_fr (stim - smooth bs)
        """
        absidx, logidx = self.get_index(stim=stim)
        return (self.idx_firing_rates(absidx, frcc))

    def firing_rates(self, trailing_type, trailing_param, fr_type=1):
        """
        calcualte the firing rates for every trial, the net frs obtained here didn't smoothen the baseline frs
        :param trailing_type: fixed_duration or proportional to duration
        :param trailing_param: specific parameter used to calcualte the
        :param fr_type:
            0: baseline, 1: stimulus, 2: stimulus - baseline (raw), other: all three in a list
        :return:
        """
        if trailing_type == "fixed_duration":
            return (self._fr_fixed_trailing(trailing_param, fr_type=fr_type))
        elif trailing_type == "proportional":
            return (self._fr_prop_trailing(trailing_param, fr_type=fr_type))
        else:
            print("Wrong parameter in firing_rates method!")
            return (None)

    def _fr_fixed_trailing(self, trailing_dur, fr_type=1):
        """
        calculate the firing rate for every trial, electrode, and every bird.
        input:
            fr_type:
                0: baseline firing rate
                1: firing rate during stimulus period + trailing dur
                2: stimulus fr - baseline fr
                3 or other values: return baseline, stimulus, net fr in a list
            trailing_dur:
                duration after stimulus offset used to calculate the firing rate
                default: no trailing
        output:
            firing rate specificed by the fr_type
        note:
            Depend on the condition that the spiketrain of each trial is centered according to the stimulus onset time.
        """
        num_items = self.spikes.shape[0]
        tmp_header = self.header.iloc[0]
        st_time = tmp_header.pre_stim
        st_index = int(st_time / tmp_header.resolution)

        baseline_frs = np.sum(self.spikes[:, :st_index], axis=1) / st_time
        if fr_type == 0:
            return (baseline_frs)
        stimulus_frs = np.zeros(num_items)
        for idx in xrange(num_items):
            tmp_header = self.header.iloc[idx]
            tmp_spiketrain = self.spikes[idx, :]
            stim_dur = tmp_header.ending_time - tmp_header.starting_time
            ed_time = st_time + stim_dur + trailing_dur
            ed_index = int(ed_time / tmp_header.resolution)
            # baseline_frs[idx] = np.sum(tmp_spiketrain[:st_index]) / st_time
            stimulus_frs[idx] = np.sum(tmp_spiketrain[st_index:ed_index]) / (stim_dur + trailing_dur)
        net_frs = stimulus_frs - baseline_frs
        fr_list = [baseline_frs, stimulus_frs, net_frs]
        if fr_type in [0, 1, 2]:
            return (fr_list[fr_type])
        else:
            return (fr_list)

    def _fr_prop_trailing(self, trailing_prop, fr_type=1):
        """
        calculate the firing rates for every trial, electrode, and every bird.
        input:
            fr_type:
                0: baseline firing rate
                1: firing rate during stimulus period + trailing dur
                2: stimulus fr - baseline fr
                3 or other values: return baseline, stimulus, net fr in a list
            trailing_dur:
                duration after stimulus offset used to calculate the firing rate
                default: no trailing
        output:
            firing rate specificed by the fr_type
        note:
            Depend on the condition that the spiketrain of each trial is centered according to the stimulus onset time.
        """
        num_items = self.spikes.shape[0]
        tmp_header = self.header.iloc[0]
        st_time = tmp_header.pre_stim
        st_index = int(st_time / tmp_header.resolution)

        baseline_frs = np.sum(self.spikes[:, :st_index], axis=1) / st_time
        if fr_type == 0:
            return (baseline_frs)
        stimulus_frs = np.zeros(num_items)
        for idx in xrange(num_items):
            tmp_header = self.header.iloc[idx]
            tmp_spiketrain = self.spikes[idx, :]
            stim_dur = tmp_header.ending_time - tmp_header.starting_time
            trailing_dur = stim_dur * trailing_prop
            ed_time = st_time + stim_dur + trailing_dur
            ed_index = int(ed_time / tmp_header.resolution)
            stimulus_frs[idx] = np.sum(tmp_spiketrain[st_index:ed_index]) / (stim_dur + trailing_dur)
        net_frs = stimulus_frs - baseline_frs
        fr_list = [baseline_frs, stimulus_frs, net_frs]
        if fr_type in [0, 1, 2]:
            return (fr_list[fr_type])
        else:
            return (fr_list)

    def batch_process_data(self, func, DAC, frcc=frcc):
        """
        calculate desired quantity for each electrode.
        :param func: function used to analyze individual electrode
        :param DAC: a dictionary storing analysis specific parameters
        :param frcc: common parameters used to calculate neural responses
        :return:
        """
        identifier_list = []
        data_list = []
        res_list = []
        for birdid in self.birdid:
            bird_data = self.get_kw_SpikeData(birdid=birdid)
            for ele_id in bird_data.electrodes:
                ele_data = bird_data.get_kw_SpikeData(electrode=ele_id)
                ele_responsivenss = ele_data.res_check(frcc).res_data[0]
                tmp_data = func(ele_data, DAC, frcc)
                identifier_list.append(birdid + separator + str(ele_id))
                res_list.append(ele_responsivenss)
                data_list.append(tmp_data)
            gc.collect()
        # print(birdid)
        return (np.asarray(identifier_list), data_list, np.asarray(res_list))

    def bird_batch_pd(self, func, DAC, frcc=frcc):
        pass

    #def _id2header(self):
        """
        add id to the header. header = birdid + "___" + electrode
        :return:
        """
        #self.header["id"] = [tmp_bird + separator + str(ele) for tmp_bird, ele in
                             #zip(self.header.birdid.values, self.header.electrode.values)]

    def filter(self, func):
        """
        Remove non-responding sites and keep only responsive sites.
        If nothing is specified, one checks whether the electrodes keep responding.
        :param func: a function handle used to test the electrode data
        :return:
        """
        non_responsive_logidxes = np.zeros(len(self.header), dtype=np.bool)
        for bird in self.birdid:
            bird_data = self.get_kw_SpikeData(birdid=bird)
            for ele in bird_data.electrodes:
                absidx, logidx = self.get_index(birdid=bird, electrode=ele)
                ele_data = self.get_idx_SpikeData(absidx, logidx)
                # responsive = func(ele_data)
                responsive = func(ele_data)
                if not responsive:
                    non_responsive_logidxes += logidx  # set non-responsive index to true
                    # raster_plot(ele_data.spikes)
        responsive_logidxes = ~non_responsive_logidxes
        return SpikeData(self.header[responsive_logidxes], self.spikes[responsive_logidxes, :])

    def site_stats(self, frcc=frcc, param=PARAM):
        """
        test whether a site is good or not.\n
        This is a minimal test about whether one electrode/neuron is responding or not. Not specific to any stimuli.
        A site is responding as long as it is responsive to at least one stimuli.
        It does NOT care whether it is an inhibitory or excitatory site.
        Returned data are in different order from res_check.
        :return:
        """
        trailing_type = frcc.trailing_type
        trailing_param = frcc.get_trailing_param(trailing_type)
        stats_test = frcc.stats_test

        # initialize container
        _stim = []
        _p_value = []
        _avg_responses = []
        _statistics = []
        for stim in self.stim:
            # print("The electrode and stimulus are %f and %f" % (electrode, stim))
            tmp_data = self.get_kw_SpikeData(stim=stim)
            tmp_responses = tmp_data.firing_rates(trailing_type, trailing_param, fr_type=2)
            if stats_test == "binomial":
                num_positive = np.sum(tmp_responses > 0)
                total = len(tmp_responses)
                p_value = scipy.stats.binom_test(num_positive, total, p=0.5)
                statistics = num_positive * 1.0 / total
            elif stats_test == "t_test":
                statistics, p_value = scipy.stats.ttest_1samp(tmp_responses, popmean=0)
            else:
                statistics, p_value = scipy.stats.wilcoxon(tmp_responses)
            _stim.append(stim)
            _p_value.append(p_value)
            _statistics.append(statistics)
            _avg_responses.append(robust_mean(tmp_responses, num_mad=param.outlier_threshold))
        tmp_data = np.vstack((_p_value, _avg_responses, _statistics))
        df = pd.DataFrame(data=tmp_data.T, index=_stim, columns=["p", "fr", "stats"])
        return (df)

    def res_check(self, frcc=frcc, fr_type=2,
                  filter_type="p_value", param=PARAM):
        """
        calculate whether an electrode responds to a stimulus
        :param stats_test: "binomial", "wilcoxon", or "t_test"
        :param stats_test: "binomial", "wilcoxon", or "t_test"
        :param fr_type:
        :param filter_type: "p_value", p_value; "abs_avg_response": absolute average_responses
        :return:
        responsivenss_df:
            structure of the responsiveness_data.
                Each column represents data from one stimulus.
                The 1st row gives the stimulus code.
                The 2nd row gives p-value.
                The 3rd row gives the statistic.
                The 4th row gives the average response to each stimulus.
        """
        trailing_type = frcc.trailing_type
        trailing_param = frcc.get_trailing_param(trailing_type)
        stats_test = frcc.stats_test
        _birdid = []
        _electrodes = []
        _responsiveness_data = []
        _quick_responsiveness_filter = []
        for bird in self.birdid:
            bird_data = self.get_kw_SpikeData(birdid=bird)
            for electrode in self.electrodes:
                _stim = []
                _p_value = []
                _statistics = []
                _avg_responses = []
                electrode_data = bird_data.get_kw_SpikeData(electrode=electrode)
                for stim in self.stim:
                    # print("The electrode and stimulus are %f and %f" % (electrode, stim))
                    tmp_data = electrode_data.get_kw_SpikeData(stim=stim)
                    tmp_responses = tmp_data.firing_rates(trailing_type, trailing_param, fr_type=fr_type)
                    if stats_test == "binomial":
                        num_positive = np.sum(tmp_responses > 0)
                        total = len(tmp_responses)
                        p_value = scipy.stats.binom_test(num_positive, total, p=0.5)
                        statistics = num_positive * 1.0 / total
                    elif stats_test == "t_test":
                        statistics, p_value = scipy.stats.ttest_1samp(tmp_responses, popmean=0)
                    else:
                        statistics, p_value = scipy.stats.wilcoxon(tmp_responses)
                    _stim.append(stim)
                    _p_value.append(p_value)
                    _statistics.append(statistics)
                    _avg_responses.append(robust_mean(tmp_responses, num_mad=param.outlier_threshold))
                _birdid.append(bird)
                _electrodes.append(electrode)
                tmp_data = np.vstack((_stim, _p_value, _statistics, _avg_responses))
                _responsiveness_data.append(np.copy(tmp_data))
                if filter_type == "p_value":
                    _quick_responsiveness_filter.append(np.max(tmp_data[1]))
                elif filter_type == "abs_avg_response":
                    _quick_responsiveness_filter.append(np.min(np.abs(tmp_data[3])))

        responsive_df = pd.DataFrame({"birdid": np.asarray(_birdid), "electrode": np.asarray(_electrodes),
                                      "res_data": _responsiveness_data,
                                      "res": _quick_responsiveness_filter})
        return (responsive_df)


class HistologyData(object):
    """
    stores histology data. can be used to separate electrodes into different categories.
    """
    def __init__(self, id_area_df, areas2include=None, area2adr_dict=None, missing="missing"):
        """
        load the dataframe to populate internal data.
        :param id_area_df: a pandas dataframe from .csv file. has the following columns.
            1. id: string representation of the bird id
            2. electrode: electrode = (channel number - 1) * 100 for MUA. SUA from the same site has the same base + 1/2/3 for SUA.
            3. histology: area from the histology.
            4. function: functional correction for sites that are not clear from histology.
            5. area: area decided based on all information. Final histology decision.
            6. histology_check (optional): flag whether the histology has been checked.
            7. function_check (optional): whether the site has been checked by independent recording files.
            8. exp (optional): name of the experiment.
            9. comment (optional): any extra comment.
        :param: area2adr_dict: a dictionary
        :param missing:
        """

        def __update_area(df, missing=missing):
            """
            determine the area based on histology and function data.
            :param df:
            :param missing: value used to fill missing values.
            :return:
            """
            area = df.histology
            if df.function != missing:  # if function specifies area, use it
                area = df.function
            if df.comment != missing:  # if comment is not empty, return as bad channel
                area = "bad"
            return area

        def __area2adr(id_area_df, area2adr_dict=None):
            """
            add a fake adaptation rate to the data so that it is compatible for old code that uses adaptation rate to separate electrodes.
            :param area2adr_dict: a dictionary whose key is area, value is an adaptation rate.
                e.g., {"L2": 1, "NCM": -1}. Then one could use threshold 0 to decide which electrode is from L2 or NCM
            :return:
            """
            id_area_df["adr"] = None
            if area2adr_dict is not None:
                for key in area2adr_dict.keys():
                    id_area_df.loc[id_area_df["area"] == key, "adr"] = area2adr_dict[key]
            return id_area_df

        def __filter(df, areas2include):
            """
            in place filter the data.
            select areas only from those listed. All other entries will be deleted.
            :param areas2include: a list stores areas to include. e.g., ["L2", "NCM"]
            :return:
            """
            logidx = np.zeros(len(df), dtype=np.bool)
            for area in areas2include:
                logidx += df["area"].values == area
            id_area_df = df[logidx]
            return id_area_df

        # fill missing value
        self.id_area_df = id_area_df.fillna(missing)
        # reset index
        self.id_area_df.reset_index(drop=True, inplace=True)

        # combine birdid and electrode number
        self.id_area_df["id"] = self.id_area_df.apply(lambda df: df.id + separator + str(df.electrode), axis=1)
        # update area based on histology, function correction, and comment
        self.id_area_df["area"] = self.id_area_df.apply(__update_area, axis=1)

        # filter the id_area_df
        if areas2include is not None:
            self.id_area_df = __filter(self.id_area_df, areas2include)

        # add fake adaptation rate for old codes
        self.id_area_df = __area2adr(self.id_area_df, area2adr_dict)
        # create dictionary for fast processing
        self.id_area_dict = df2dict(self.id_area_df, key="id", val="area")
        self.id_adr_dict = df2dict(self.id_area_df, "id", "adr")

    def get_area(self, id):
        """
        get the area/histology for the unique id.
        :param id: a string that combines birdid and electrode number.
        :return:
        """
        split_results = id.split(separator)
        if len(split_results) != 2:
            raise ValueError("Wrong ID name, do you use '___' to join birdid and electrode number? ")
        else:
            bird, ele = split_results
            ele_int = int(ele)
            ele = int(ele_int / mua_sua_distinguisher) * mua_sua_distinguisher
            new_id = bird + separator + str(ele)
            if new_id not in self.id_area_dict:
                return None
            else:
                return self.id_area_dict[new_id]


def raster_plot(spike_matrix, resolution=resolution, figsize=(5, 4), new_figure=True, linewidth=0.9):
    """
    input:
        spike_matrix: a integer/double/boolen numpy matrix with 1 indicating spike and
                        0 indicating no spike
                      each row represents one trial
        resolution: discrete resolution / bin_size used descretize spike timing data,
                    in unit of seconds
        figsize: figure size
        new_figure: whether create a new figure
        linewidth: thickness of the raster plot
    output:
        figure handle of the raster plot
    """
    mpl.rcParams['lines.linewidth'] = linewidth  # set the linewidth for raster plot
    fig = None
    if new_figure:
        fig = plt.figure(figsize=figsize)
    line_height = 0.9
    for iii, spikes in enumerate(spike_matrix):
        timing = np.flatnonzero(spikes) * resolution
        plt.vlines(timing, iii,
                   iii + line_height)
    plt.xlim(0, spike_matrix.shape[1] * resolution)
    plt.ylim(0, spike_matrix.shape[0])
    plt.xlabel("time");
    plt.ylabel("trial/neuron index")
    plt.gca().invert_yaxis()
    mpl.rcParams['lines.linewidth'] = 1  # restore to default
    if new_figure:
        return fig


def heat_plot(spike_matrix, step_size=int(0.01 / resolution), cmap="jet", cmap_limit=None,
              figsize=(5, 4), new_figure=True, aspect="auto", outlier_removal=False):
    """
    heat plot representation of raster plot
    input:
        spike_matrix: a integer/double/boolen numpy matrix with 1 indicating spike and
                        0 indicating no spike
                      each row represents one trial
        resolution: discrete resolution / bin_size used descretize spike timing data,
                    in unit of seconds
        figsize: figure size
    output:
        figure handle of the raster plot
    """
    fig = None
    if new_figure:
        fig = plt.figure(figsize=figsize)
    if step_size != 1:
        spike_matrix = step_sum(spike_matrix, step_size, axis=1)

    if outlier_removal:
        avg = scipy.stats.trim_mean(spike_matrix, 0.3)
        std = sm.robust.mad(sm.robust.mad(spike_matrix))
        tmp_max = avg + std * 3.5
        tmp_min = avg - std * 3.5
        spike_matrix = np.clip(spike_matrix, tmp_min, tmp_max)

    if cmap_limit is None:
        plt.imshow(spike_matrix, origin="upper", cmap=cmap, aspect=aspect,
                   interpolation='nearest')
    else:
        plt.imshow(spike_matrix, origin="upper", cmap=cmap,
                   vmin=cmap_limit[0], vmax=cmap_limit[1], aspect=aspect,
                   interpolation='nearest')
    # plt.gca().invert_yaxis()
    plt.colorbar()
    if new_figure:
        return fig


def quick_category_plot(df, jitter=0.25, new_figure=False, figsize=(6, 4),
                        size=5, alpha=1):
    """
    make a quick category plot
    :param df: df is a simple dataframe whose columns are measurements from different conditions
    :param jitter: jitter size of the strip/dot plot
    :param new_figure:
    :param figsize:
    :return:
    """
    fig = None
    if new_figure:
        fig = plt.figure(figsize=figsize)
    sns.stripplot(data=df, jitter=jitter, size=size, alpha=alpha)
    sns.boxplot(data=df, fliersize=1e-10, color="white")
    return (fig)


def quick_long_plot(df, x, y, jitter=0.25, new_figure=False, figsize=(6, 4),
                    size=5, alpha=1, order=None):
    """
    make a quick category plot
    :param df: df is a simple dataframe whose columns are measurements from different conditions
    :param jitter: jitter size of the strip/dot plot
    :param new_figure:
    :param figsize:
    :return:
    """
    fig = None
    if new_figure:
        fig = plt.figure(figsize=figsize)
    if order is None:
        sns.stripplot(data=df, x=x, y=y, jitter=jitter, size=size, alpha=alpha)
        sns.boxplot(data=df, x=x, y=y, fliersize=1e-10, color="white")
    else:
        sns.stripplot(data=df, x=x, y=y, jitter=jitter, size=size, alpha=alpha, order=order)
        sns.boxplot(data=df, x=x, y=y, fliersize=1e-10, color="white", order=order)
    return (fig)


def categorical_scatterplot(x=None, y=None, categories=None, xlabel=None, ylabel=None, category_label=None,
                            figsize=None):
    """
    make a scatter plot, different categories appear as different color and shape
    :param x:
    :param y:
    :param catories:
    :param figsize:
    :return:
    """
    if x is None:
        x = np.arange(1, len(y) + 1, 1)
    #     categories = [str(item) for item in categories]
    flatpal = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    if xlabel is None:
        xlabel = "x"
    if ylabel is None:
        ylabel = "y"
    if category_label is None:
        category_label = "category"

    df = pd.DataFrame({xlabel: x, ylabel: y, category_label: categories})
    if figsize is not None:
        with sns.color_palette(flatpal):
            fig = plt.figure(figsize=figsize)
            sns.scatterplot(data=df, x=xlabel, y=ylabel, hue=category_label, style=category_label, palette="Set1")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig
    else:
        with sns.color_palette(flatpal):
            sns.scatterplot(data=df, x=xlabel, y=ylabel, hue=category_label, style=category_label, palette="Set1")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return None


def quick_cdf_plot(df, new_figure=False, linewidth=1.5, figsize=(6, 4), color="black"):
    """
    plot empirical cumulative frequency function
    :param df:
    :param new_figure:
    :param linewidth:
    :param figsize:
    :param color:
    :return:
    """
    fig = None
    if new_figure:
        fig = plt.figure(figsize=figsize)
    num_bins = len(df)
    for col in df.columns:
        hist_kws = {"histtype": "step", "linewidth": linewidth, "cumulative": True, "alpha": 1}
        valid_values = df[col][np.logical_not(np.isnan(df[col]))]
        if len(df.columns) == 1:
            sns.distplot(valid_values, bins=num_bins + 1, hist_kws=hist_kws, kde=False, norm_hist=True,
                         color=color)
        else:
            sns.distplot(valid_values, bins=num_bins + 1, hist_kws=hist_kws, kde=False,
                         norm_hist=True)
        # plt.hist(df[col], bins=num_bins, histtype="step", cumulative=True,
        #          linewidth=linewidth, normed=True)
    plt.legend(df.columns)
    plt.ylabel("Cumulative Frequency")
    return (fig)


def pretty_specgram(y, fs, fft_size=512, figsize=(8, 6),
                    cmap="jet", kill_off=True):
    def kill_off_signal(data):
        data_size = data.shape
        data = data[:]
        avg = np.median(data)
        std = np.std(data)
        lb = avg + 0.25 * std
        ub = avg + 10 * std
        data = np.clip(data, lb, ub)
        return (np.reshape(data, data_size))

    # shift_time determines the overlapping between FFT windows
    fft_time_shift_seconds = 0.002

    y = np.asarray(y, dtype=np.double)

    # audio times
    # audio_times = np.arange(0, len(y)) * 1.0 / fs
    noverlap = fft_size - np.floor(fs * fft_time_shift_seconds)

    # calculate spectrogram
    freqs, t, spectrogram = signal.spectrogram(y, fs, window=signal.hamming(fft_size),
                                               nperseg=fft_size, noverlap=noverlap)
    spectrogram = np.abs(spectrogram)
    log_spectrogram = np.log(spectrogram + np.finfo(np.float64).eps)

    # bound the signals for display purpose
    if kill_off:
        log_spectrogram = kill_off_signal(log_spectrogram)
    fh = plt.figure(figsize=figsize)
    plt.pcolormesh(t, freqs / 1000, log_spectrogram, cmap=cmap)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [sec]')
    return (fh, log_spectrogram)


def smooth_baseline(frs, frcc):
    """
    smooth the baseline firing rate
    :param trials:
    :param frs:
    :param smooth_type:
        avg: average of all baseline
        robust_avg: average of baseline after excluding extreme values (20% of the )
    :return:
    """
    smooth_type = frcc.bs_smooth_type
    smooth_mg = frcc.bs_smooth_mg
    outlier_factor = frcc.bs_outlier_factor
    prop_cut = frcc.bs_prop_out

    num_trials = len(frs)
    if smooth_type == "raw":
        return (frs)
    elif smooth_type == "avg":
        return (np.ones(num_trials) * np.mean(frs))
    elif smooth_type == "robust_avg":
        return (np.ones(num_trials) * scipy.stats.trim_mean(frs, proportiontocut=prop_cut))
    elif smooth_type == "linreg":
        trials = np.arange(len(frs))
        p = np.polyfit(trials, frs, deg=1, full=False)
        fitted_vals = np.polyval(p, trials)
        return (fitted_vals)
    elif smooth_type == "robust_linreg":
        outlier_logidx = np.abs(frs - scipy.stats.trim_mean(frs, prop_cut)) > \
                         outlier_factor * np.std(frs, ddof=1)
        inlier_logidx = np.logical_not(outlier_logidx)
        trials = np.arange(len(frs))
        inlier_trials = trials[inlier_logidx]
        inlier_frs = frs[inlier_logidx]
        p = np.polyfit(inlier_trials, inlier_frs, deg=1, full=False)
        fitted_vals = np.polyval(p, trials)
        return (fitted_vals)
    elif smooth_type == "local_smooth":
        local_smoothed_fr = np.zeros(num_trials)
        for iii in range(num_trials):
            st_idx = max([iii - smooth_mg, 0])
            ed_idx = iii + smooth_mg + 1
            local_smoothed_fr[iii] = scipy.stats.trim_mean(frs[st_idx:ed_idx],
                                                           proportiontocut=prop_cut)
        return (local_smoothed_fr)
    else:
        print("Wrong smooth_type! Valid smooth_type should be avg, robust_avg, linreg, robust_linreg, local_smooth")
        return (None)


def dur_extract(cse):
    """
    extract stimulus duration from the header information
    input:
          cse: numpy array of stimulus code, stimulus starting time, stimulus ending time
    output:
          (stim_code, stim_dur): numpy array of stimulus code and corresponding duration
    """
    stim_codes = cse[:, 0]
    unq_stim_code = np.unique(stim_codes)
    duration = np.diff(cse[:, 1:], axis=1)
    stim_dur = [np.mean(duration[code == stim_codes]) for code in unq_stim_code]
    return np.vstack((unq_stim_code, stim_dur))


def spikes2fr(spikes, time_range=None, resolution=resolution):
    """
    calculate firing rates during specific time range
    input:
         spikes: a boolen numpy array of spikes
         resolution: sampling_frequency
         time_range: a numpy array [starting_time, ending_time]
    output:
         f_rate
    """
    if time_range is not None:
        index_range = np.asarray(time_range) / resolution
        index_range = index_range.astype(np.int64)
        spikes = spikes[:, index_range[0]: index_range[1]]

    duration = spikes.shape[1] * resolution
    spike_counts = np.sum(spikes, axis=1)
    f_rate = spike_counts / duration
    return (f_rate)


def adaptation_rate(response, trial=None, robust=False):
    """
    calculate the adaptation rate using David's method
    input:
        response: 1-d numpy array
        trial: trial number to regress on, 1-d numpy array
    output:
        adr: David's adaptation rate
    """
    if robust:
        adr = _robust_adaptation_rate(response, trial)
        return (adr)
    if trial is None:
        trial = np.arange(1, response.shape[0] + 1)
    else:
        response = response[trial]
    trial = np.vstack((trial, np.ones(trial.shape))).T
    a, b = np.linalg.lstsq(trial, response)[0]
    norm_factor = np.mean(response)
    adr = a / norm_factor
    return adr


def _robust_adaptation_rate(response, trial=None, prop_out=robust_adr_propout):
    """
    calculate the adaptation rate using robust regression from statsmodels
    input:
        response: 1-d numpy array
        trial: trial number to regress on, 1-d numpy array
    output:
        adr: David's adaptation rate
    """
    if trial is None:
        trial = np.arange(1, response.shape[0] + 1)
    else:
        response = response[trial]
    trial = np.vstack((trial, np.ones(trial.shape))).T
    res = sm.RLM(response, trial,
                 M=sm.robust.norms.HuberT()).fit()  # using Huber's function to deal with potential outlier
    # norm_factor = np.median(response) # median is more resistent to outlier
    robust_norm_factor = scipy.stats.trim_mean(response, prop_out)  # trimmed mean is robust and efficient
    adr = res.params[0] / robust_norm_factor
    return adr


def birdid2str(birdid):
    """
    convert numeric bird name into string representation
    input:
        birdid: number from literal letter to number convert (e.g., A -> 65)
    output:
        birdname:
    """
    birdstr = str(birdid)
    birdname = chr(int(birdstr[:2])) + chr(int(birdstr[2:4])) + birdstr[4:]
    return (birdname)


def str2birdid(birdname):
    """
    convert bird name to numeric representation
    :param birdname: bird name as a string
    :return:
    """
    bird_color = birdname[:2]
    bird_code = str(birdname[2:])
    bird_numeric_color = "".join([str(ord(item)) for item in bird_color])
    birdid = int(bird_numeric_color + bird_code)
    return (birdid)


def running_avg(data, smooth_mg, axis=1):
    """
    input:
        data: a 1D or 2D numpy array to be smoothed
            if 2D: it's smoothed across the specified axis
        smooth_mg: bin_size to perform smoothing
    output:
        smooth_data: smoothed vector of the same size as vec
    note: beginning and ending of smooth_vec may act strange because the padding of zeros
    """
    kernel = np.ones(smooth_mg, dtype=np.float64) / smooth_mg
    smooth_data = None
    if len(data.shape) == 1:
        smooth_data = np.convolve(data, kernel, mode="same")
        return (smooth_data)
    elif len(data.shape) == 2:
        if axis == 0:
            data2process = np.transpose(data)
            smooth_data = [np.convolve(tmp_data, kernel, mode="same") for tmp_data in data2process]
            smooth_data = np.transpose(np.array(smooth_data))
        elif axis == 1:
            data2process = data
            smooth_data = [np.convolve(tmp_data, kernel, mode="same") for tmp_data in data2process]
            smooth_data = np.array(smooth_data)
        return (smooth_data)
    else:
        print("Error!")
        return None


def step_sum(data, step_size, axis=1):
    """
    input:
        data: a 1D or 2D numpy array to be smoothed
            if 2D: it's smoothed across the specified axis
        step_size: step used to sum over
        axis: axis to step over
    output:
        step_data:
    """
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
        data = ___step_sum_2d(data, step_size)
        return (data[0, :])
    elif len(data.shape) == 2:
        if axis == 1:
            return (___step_sum_2d(data, step_size))
        elif axis == 0:
            data = np.transpose(data)
            return (___step_sum_2d(data, step_size).T)
    else:
        print("Error!")
        return None

def list_files(regexp=None):
    """
    find files in the selected directories and return their absolute path
    :param: regexp, a regular expression string used to filter files.\n
        e.g., ".*\.npz" will only list files with .npz extension.\n
              ".*\.mat" will only list files with .mat extension.\n
    :return:
    """
    import re
    import Tkinter
    from tkFileDialog import askopenfilename, askdirectory
    from os import walk

    # load the data from matlab files using gui
    root = Tkinter.Tk()
    path_name = askdirectory(parent=root, initialdir="./")
    root.withdraw()  # close the main root window
    # path = "C:\\Users\\md\\Dropbox\\Lab_Data\\2015_NCM_syllable_surprisal\\Raw_data\\YW570\\"
    fn_list = []
    for (dirpath, dirnames, filenames) in walk(path_name):
        if regexp is None:
            filenames = [dirpath + "/" + item for item in filenames]
            fn_list.extend(filenames)
        else:
            filenames = [dirpath + "/" + item for item in filenames if re.search(regexp, item)]
            fn_list.extend(filenames)
    return (fn_list)


def get_pathname(data_dir="D:\\Google_Drive\\Lab_Data\\"):
    # from Tkinter import *
    import Tkinter
    from tkFileDialog import askopenfilename

    # load the data from matlab files using gui
    root = Tkinter.Tk()
    path_name = askopenfilename(parent=root, initialdir=data_dir)
    root.withdraw()  # close the main root window
    return (path_name)


def load_mat_file(data_dir="D:\\Google_Drive\\Lab_Data\\"):
    # from Tkinter import *
    import Tkinter
    from tkFileDialog import askopenfilename
    import scipy.io

    # load the data from matlab files using gui
    root = Tkinter.Tk()
    path_name = askopenfilename(parent=root, initialdir=data_dir)
    root.withdraw()  # close the main root window
    # path = "C:\\Users\\md\\Dropbox\\Lab_Data\\2015_NCM_syllable_surprisal\\Raw_data\\YW570\\"
    # filename = "matrix_MDYW570_Kai_CanarySongContext_6scISI_.mat"
    data = scipy.io.loadmat(path_name)
    return (data, path_name)


def _mat2npz(path_name, electrode_adjustment, convert_birdid):
    """
    load .mat data and convert it into header & spikes.
    :param path_name:
    :param electrode_adjustment:
    :param convert_birdid:
    :return:
    """
    data = scipy.io.loadmat(path_name)
    # separate the header and spike matrix
    header = data["header"]
    spike_data = data["spiketrains"].astype(np.bool)
    header_names = ["birdid", "electrode", "trial",
                    "stim", "starting_time", "ending_time",
                    "pre_stim", "after_stim", "resolution"]
    header_df = pd.DataFrame(data=header, columns=header_names)
    header_df[["electrode", "trial", "stim"]] = header_df[["electrode", "trial", "stim"]].astype(int)
    header_df["electrode"] -= electrode_adjustment  # electrode number from 1
    if convert_birdid:
        header_df["birdid"] = [birdid2str(int(bird)) for bird in np.asarray(header_df.birdid)]

    # load stimulus waveform information
    stim = np.squeeze(data["stim_codes"])
    stim_sr = data["stim_sr"][0][0]
    stim_waveforms = data["stim_waveforms"]
    # 1st column: birdid, 2nd column: stim_code, 3rd column: sampling rate, 4th column: NaN, 5th and later: stimulus waveforms
    stim_arry = []
    bird = int(header[0, 0])
    for i in range(len(stim)):
        tmp_array = [bird, stim[i], stim_sr, np.nan] + list(stim_waveforms[i])
        stim_arry.append(tmp_array)
    stim_arry = np.asarray(stim_arry)

    return (header_df, spike_data)


def load_mat_data(data_dir="D:\\Google_Drive\\Lab_Data\\",
                  electrode_adjustment=0, convert_birdid=False):
    """
    input:
        1) select .mat data file (structured data from matlab script)
        2) default data directory
    output:
        header_df: pandas dataframe with trial information
        spike_data: correspondong 2d numpy array, 1 indicates spike
    note:
        the spiketrain of each trial is centered according to the stimulus onset time
    """
    import Tkinter
    from tkFileDialog import askopenfilename

    # load the data from matlab files using gui
    root = Tkinter.Tk()
    path_name = askopenfilename(parent=root, initialdir=data_dir)
    root.withdraw()  # close the main root window
    header_df, spike_data = _mat2npz(path_name, electrode_adjustment, convert_birdid)
    return (header_df, spike_data, path_name)


def batch_mat2npz(npz_filename, directory="C:\\Users\\md\\Dropbox\\Lab_Data\\", electrode_adjustment=0, convert_birdid=False):
    """
    load a list of matlab matrix files in the directory, concatenate them, and save them as .npz file
    :param npz_filename:
    :param directory:
    :param electrode_adjustment:
    :param convert_birdid:
    :return:
    """
    import Tkinter
    from tkFileDialog import askdirectory

    # load the data from matlab files using gui
    root = Tkinter.Tk()
    path_name = askdirectory(initialdir=directory, title='select where matrix files are located')
    root.withdraw()  # close the main root window
    kw = "*.mat"

    files = glob.glob(path_name + "/" + kw)
    path_length = len(path_name) + 1

    header_list = []
    spikes_list = []
    fn_list = []
    pre_stims = []  # store pre_stim from each data files
    stim_waveforms = []
    stim_max_ndpts = 0  # number of data points in the longest stimulus

    for fn in files:
        # add recording names to the header for backup use
        recording_fn = fn[path_length:]
        print(recording_fn)
        header, spikes = _mat2npz(fn, electrode_adjustment, convert_birdid)
        fn_list.extend([recording_fn] * len(header))
        header_list.append(header)
        spikes_list.append(spikes)
        pre_stims.append(np.mean(header.pre_stim.values))

        # add stim_waveforms
        #David Edit
        ### stim_waveforms.append(stims)
        ##if stims.shape[1] > stim_max_ndpts:
            ###stim_max_ndpts = stims.shape[1]

    # pad short stim with NaN so that all stimuli have the same length
    ##for i in range(len(stim_waveforms)):
      #  if stim_waveforms[i].shape[1] >= stim_max_ndpts:
       #     continue
        #else:
        #    num_pads = stim_max_ndpts - stim_waveforms[i].shape[1]
         #   stim_waveforms[i] = np.pad(stim_waveforms[i], ((0, 0), (0, num_pads)), mode="constant", constant_v#alues=np.nan)
    #stim_waveforms = np.vstack(stim_waveforms)

    # test whether pre_stim are the same (only applicable for randomized ISI)
    ncols = []
    if len(np.unique(pre_stims)) == 1:
        # if all pre_stims are the same, do nothing
        ncols = [spikes.shape[1]]

    else: # if pre_stim is different, store the shortest one
        shortest_pre_stim = np.min(pre_stims)
        resolution = np.mean(header.resolution.values)
        cut_idx = int(shortest_pre_stim / resolution)
        # cut the extra baseline from spikes by slicing

        for idx in range(len(spikes_list)):
            tmp_st_idx = int(np.mean(header_list[idx].pre_stim.values) / resolution)
            spikes_list[idx] = spikes_list[idx][:, max(tmp_st_idx - cut_idx, 0):]  # change spikes list in place
            # change the pre_stim in all headers
            header_list[idx]["pre_stim"] = np.ones(len(header_list[idx])) * shortest_pre_stim
            ncols.append(spikes_list[idx].shape[1])

    header_all = pd.concat(header_list)
    filename_array = np.asarray(fn_list)

    ncols = np.asarray(ncols)

    # test whether all spikes have the same size
    if len(np.unique(ncols)) == 1:
        spike_all = np.vstack(spikes_list)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Spikes data have different columns.")
        print(ncols)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        min_length = np.min(ncols)
        for idx in range(len(spikes_list)):
            spikes_list[idx] = spikes_list[idx][:, :min_length]
        spike_all = np.vstack(spikes_list)
        print("Final spike shape:", spike_all.shape)
    if (type(header_all["birdid"].values[0]) is np.str):
        header_all["birdid"] = [str2birdid(bird) for bird in header_all.birdid.values]

    np.savez_compressed(path_name + "/" + npz_filename, header=header_all.values, spikes=spike_all, filenames=filename_array)


def save_npz_data(filename, header, spikes, stims):
    """
    save the header and spikes into one single .npz file
    :param filename:
    :param header:
    :param spikes:
    :return:
    """
    if (type(header["birdid"].values[0]) is np.str):
        header["birdid"] = [str2birdid(bird) for bird in header.birdid.values]
    np.savez_compressed(filename, header=header.values, spikes=spikes)


def load_npz_data(filename, convert_birdid, param=PARAM):
    ''''
    load the saved npz data for analysis
    :param filename:
    :param convert_birdid:
    :return:
    def _array2dict(stims, param):
    
        convert stim waveforms from array format into dictionary for easier calculation.
        :param stims:
        :return: a nested dictionary,
            dict{birdid: {stim_code: waveforms, sampling_rate: sr} }
        birdids = np.asarray([birdid2str(int(bird)) for bird in stims[:, 0]])
        codes = [int(item) for item in stims[:, 1]]
        srs = [int(item) for item in stims[:, 2]]
        stim_dict = {}
        # loop through each bird
        for bird in np.unique(birdids):
            stim_dict[bird] = {} # add that bird to the dictionary
            # find the stimulus codes from that bird
            tmp_idxes = np.flatnonzero(birdids==bird)
            # loop through every stimulus code and add stimulus waveforms into the sub-dictionary
            for i in tmp_idxes:
                code = codes[i]
                tmp_sr = srs[i]
                tmp_waveform = stims[i, param.stim_waveform_start_idx:]
                stim_dict[bird][code] = tmp_waveform[~np.isnan(tmp_waveform)]
                stim_dict[bird]["sr"] = tmp_sr
        return stim_dict
    '''
    data = np.load(filename)
    filename_flag = "filenames" in data.keys()
    header_df, spike_data = data["header"], data["spikes"]

    spike_data = spike_data.astype(np.bool)
    header_names = ["birdid", "electrode", "trial",
                    "stim", "starting_time", "ending_time",
                    "pre_stim", "after_stim", "resolution"]
    header_df = pd.DataFrame(data=header_df, columns=header_names)
    header_df[["electrode", "trial", "stim"]] = header_df[["electrode", "trial", "stim"]].astype(int)
    #stim_waveforms = data["stim_waveforms"]
    #stim_dict = _array2dict(stim_waveforms, param)
    if convert_birdid:
        # header_df["birdid"] = [birdid2str(int(bird)) for bird in np.asarray(header_df.birdid)]
        header_df["birdid"] = header_df["birdid"].apply(lambda bird: birdid2str(int(bird)))
    if filename_flag:
        recording_filenames = data["filenames"]
        header_df["filename"] = recording_filenames  # add recording filenames to the header
    return (header_df, spike_data)


def load_NpzData2SpikeData(filename, convert_birdid):
    """
    load .npz data and convert into SpikeData object
    :param filename:
    :param convert_birdid:
    :return:
    """
    header, spikes, stim_dict = load_npz_data(filename, convert_birdid=convert_birdid)
    if stim_dict is not None:
        return SpikeData(header, spikes)
    else:
        return SpikeData(header, spikes)


def ignore_nan_test(xxx, yyy=None, test_func=None, paired=False):
    """
    perform statistical test after removing missing values
    :param xxx: numpy array containing nan
    :param yyy:
    :param test_func: statistical test function
    :return:
    """
    if paired:
        if yyy is not None:
            new_x = []
            new_y = []
            for itemx, itemy in zip(xxx, yyy):
                if np.isnan(itemx) or np.isnan(itemy):
                    continue
                else:
                    new_x.append(itemx)
                    new_y.append(itemy)
            print("Number of samples from x and y are: %d & %d" % (len(new_x), len(new_y)))
            return (test_func(new_x, new_y), len(new_x), len(new_y))
        else:
            new_x = [item for item in xxx if not np.isnan(item)]
            print("Number of samples: %d" % (len(new_x)))
            print(test_func(new_x))
            return (test_func(new_x), len(new_x))
    else:
        if yyy is not None:
            new_x = [item for item in xxx if not np.isnan(item)]
            new_y = [item for item in yyy if not np.isnan(item)]
            print("Number of samples from x and y are: %d & %d" % (len(new_x), len(new_y)))
            print(test_func(new_x, new_y))
            return (test_func(new_x, new_y), (len(new_x), len(new_y)))
        else:
            new_x = [item for item in xxx if not np.isnan(item)]
            print("Number of samples: %d" % (len(new_x)))
            return (test_func(new_x), len(new_x))


# extra functions

def outlier_detection(data, num_mad=4, propout=0.25):
    """
    TO BE TESTED!!!1
    detect outliers based on the trimmed mean and median absolute deviation.
    :param data:
    :param propout:
    :param num_mad:
    :return:
    """
    if len(data.shape) == 1:
        avg = scipy.stats.trim_mean(data, propout)
        std = sm.robust.mad(data, c=0.6744897501960817)
        tmp_max = avg + std * num_mad
        tmp_min = avg - std * num_mad
        logidx = np.logical_and(data > tmp_min, data < tmp_max)
        data = np.clip(data, tmp_min, tmp_max)
        return (data, logidx)
    elif len(data.shape) == 2:
        avg = scipy.stats.trim_mean(data, propout, axis=None)
        std = sm.robust.mad(sm.robust.mad(data, c=0.6744897501960817), c=0.6744897501960817)
        tmp_max = avg + std * num_mad
        tmp_min = avg - std * num_mad
        data = np.clip(data, tmp_min, tmp_max)
        return (data, None)
    else:
        raise ValueError("Wrong input data!!!")


# helper functions
def df2dict(df, key, val):
    """
    convert a dataframe into a dictionary for fast mapping
    :param df:
    :param key:
    :param val:
    :return:
    """
    mydict = {}
    for idx in range(len(df)):
        tmp_df = df.iloc[idx]
        mydict[tmp_df[key]] = tmp_df[val]
    return (mydict)


def reshape_array(array, shape, dtype=np.double):
    """
    Note: array has to be np.double, it behave differently if it is int.
    if size of array is smaller than specified by shape, add nans to the end.\n
    if size of array is bigger than specificed by shape, remove extra elements. \n
    :param array: 2d numpy array
    :param shape:  target shape of the array
    :return: array shaped
    """
    row_diff, col_diff = np.asarray(shape) - array.shape

    # if same shape, return original array
    if row_diff == 0 and col_diff == 0:
        return array.astype(dtype=dtype)

    # if both dimension of the array is smaller, adjust both of them
    elif row_diff >= 0 and col_diff >= 0:
        new_arr = np.pad(array.astype(dtype=dtype), ((0, row_diff), (0, col_diff)),
                         mode="constant", constant_values=(np.nan, np.nan))
        return new_arr

    # if array has less rows, but more columns
    elif row_diff >= 0 and col_diff < 0:
        array = array[:, :shape[1]]  # shrink number of columns
        new_arr = np.pad(array.astype(dtype=dtype), ((0, row_diff), (0, 0)),
                         mode="constant", constant_values=(np.nan, np.nan))
        return new_arr

    # if array has more rows but less columns
    elif row_diff < 0 and col_diff >= 0:
        array = array[:shape[0], :]
        new_arr = np.pad(array.astype(dtype=dtype), ((0, 0), (0, col_diff)),
                         mode="constant", constant_values=(np.nan, np.nan))
        return new_arr

    # if array has both more rows and columns
    elif row_diff < 0 and col_diff < 0:
        array = array[:shape[0], :]
        array = array[:, :shape[1]]
        return array

    else:
        raise ValueError("Code cannot handle this case!")


def electrodes_mapping(birdid, master, depth=[np.nan, np.nan], adjustment=0):
    """
    map each electrode from a bird to a 3 dimension space with Y-point as origin.\n
    absolute distance is in micrometer.\n
    For the relative position/coordinate, the most top-posterior (dorsal-caudal) electrode is set as the 0 point. As the electrode moves down and anterior, the value increase.
    e.g., given a birdid, electrode list, adjustment (left/right, starting index), master computer hemisphere, depth information.\n
        produce a dictionary, key is the birdid + electrode number, value is a tuple (relative depth, relative anterior, recorded depth, recorded anterior, recorded lateral, left/right).\n
        assume that electrode 1 to 16 are from the left hemisphere, 17 to 32 are from the right hemisphere.
    :param birdid: name of the bird, two letters followed three numbers. e.g., "VI001".
    :param master: "left" or "right".
    :param depth: depth from left and right hemisphere.
    :param adjustment: electrode number adjustment.
    :return:
    """

    spacing = 200.0  # micro-meter
    constant = 100

    left_keys = np.arange(1, 17, 1)
    right_keys = left_keys + 16
    keys = np.concatenate((left_keys, right_keys)) * constant + adjustment # multiple by 100 so that it is consistent with new naming
    keys = [birdid + separator + str(ele) for ele in keys]
    # if master on the left, 13 is 0 point (upper posterior)
    if master == "left":

        left_vals = np.asarray([
            [3, 0],  # 1
            [3, 1],  # 2
            [3, 2],  # 3
            [3, 3],  # 4
            [2, 0],  # 5
            [2, 1],  # 6
            [2, 2],  # 7
            [2, 3],  # 8
            [1, 0],  # 9
            [1, 1],  # 10
            [1, 2],  # 11
            [1, 3],  # 12
            [0, 0],  # 13
            [0, 1],  # 14
            [0, 2],  # 15
            [0, 3]  # 16

        ])
        left_pos = np.vstack((depth[0] - (3 - left_vals[:, 0]) * spacing, left_vals[:, 1] * spacing)).T
        right_vals = left_vals
        right_pos = np.vstack((depth[1] - (3 - right_vals[:, 0]) * spacing, right_vals[:, 1] * spacing)).T

    # if master on the right, 16 is 0 point (doral-cadual)
    elif master == "right":

        left_vals = np.asarray([
            [3, 3],  # 1
            [3, 2],  # 2
            [3, 1],  # 3
            [3, 0],  # 4
            [2, 3],  # 5
            [2, 2],  # 6
            [2, 1],
            [2, 0],
            [1, 3],
            [1, 2],
            [1, 1],
            [1, 0],
            [0, 3],
            [0, 2],
            [0, 1],
            [0, 0]
        ])
        right_vals = left_vals
        left_pos = np.vstack((depth[0] - left_vals[:, 0] * spacing, left_vals[:, 1] * spacing)).T
        right_pos = np.vstack((depth[1] - right_vals[:, 0] * spacing, right_vals[:, 1] * spacing)).T

    rel_pos = np.concatenate((left_vals, right_vals))
    abs_pos = np.concatenate((left_pos, right_pos))

    vals = np.hstack((rel_pos, abs_pos))

    mapping = {}

    for idx, key in enumerate(keys):
        mapping[key] = vals[idx]

    return (mapping)


def rough_histology(coordinates):
    """
    estimate the neuroanatomical position of the electrodes based on coordinates from electrode mapping.
    :param coordinates:
    :return: "NCM", "L2", "MIDDLE"
    """
    depth_threshold = 2
    ap_threshold = 1
    if coordinates[0] >= depth_threshold and coordinates[1] <= ap_threshold:
        return "NCM"

    elif coordinates[0] == 1 and coordinates[1] == 0:
        return "NCM"

    elif coordinates[0] == 2 and coordinates[1] == 3:
        return "L2"

    elif coordinates[0] < depth_threshold and coordinates[1] > ap_threshold:
        return "L2"

    else:
        return "MIDDLE"


def info2histology(ids, masters, depths, adjustments):
    # loop through each bird and generate a pandas DF
    bird_ids = []
    locs = []
    coordinates = []
    for i in range(len(ids)):
        mydict = electrodes_mapping(ids[i], masters[i], depths[i], adjustments[i])

        # loop through mydict to generate locations
        for key in mydict.keys():
            bird_ids.append(key)
            tmp_loc = rough_histology(mydict[key])
            locs.append(tmp_loc)
            coordinates.append(mydict[key].astype(np.int))

    df = pd.DataFrame({"id": bird_ids, "area": locs, "coordinate": coordinates})
    return df


def robust_mean(data, num_mad=3.0, propout=0.25, axis=1):
    """
    calculate mean after removing outliers based on Median Absolute Deviation estimated standard deviation.
    :param data:
    :param num_mad: threshold for outlier detection, larger value means outlier values will be more extreme
    :param propout:
    :return:
    """
    data = np.asarray(data)

    def ___robust_mean(arr, num_mad, propout):
        logidx = ~np.isnan(arr)
        arr = arr[logidx]
        if len(arr) == 0:
            return np.nan
        if np.all(arr == 0):
            return 0.0
        avg = scipy.stats.trim_mean(arr, propout)
        std = sm.robust.mad(arr, c=0.6744897501960817)
        tmp_max = avg + std * num_mad
        tmp_min = avg - std * num_mad
        logidx = np.logical_and(arr > tmp_min, arr < tmp_max)
        arr = arr[logidx]
        if len(arr) == 0:
            return np.nan
        return (np.mean(arr))

    if len(data.shape) == 1:
        return ___robust_mean(data, num_mad=num_mad, propout=propout)

    else:
        if axis == 1:
            results = np.array([___robust_mean(tmp_arr, num_mad=num_mad, propout=propout) for tmp_arr in data])
            return results
        elif axis == 0:
            results = np.array([___robust_mean(tmp_arr, num_mad=num_mad, propout=propout) for tmp_arr in data.T])
            return results


## helper functions

def ___step_sum_2d(arr, bin_size):
    """
    calculate the step sum of a matrix along the 2nd axis. sum of equally spaced columns.
    :param arr:
    :param bin_size:
    :return:
    """
    nr, nc = arr.shape
    new_nc = int(nc / bin_size)
    sub_arr = arr[:, : new_nc * bin_size]
    new_arr = np.reshape(sub_arr, (nr, new_nc, bin_size), order="C")
    return (np.sum(new_arr, axis=-1))


def mad(arr, c=0.6744897501960817):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        c: normalize mad to approximate standard deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med)) / c


def binomial_test(x, p):
    """
    scipy binomial test with modified input data. calculate based on how many values are larger than 0.
    :param x: a list of measurements
    :param p:
    :return:
    """
    n_successes = np.sum(x > 0)
    n = len(x)
    pval = scipy.stats.binom_test(n_successes, n=n, p=p)
    return (n_successes * 1.0 / n, pval)

def sua2channel(sua_int):
    """
    convert a SUA number to MUA so that its location can be retrieved.
    only work for Python 2.7
    :param sua_int: numeric representation of a SUA.
    :return: numeric representation of the corresponding channel.
    """
    return int((sua_int / mua_sua_distinguisher) * mua_sua_distinguisher)


def trial_check(ele_data, frcc=frcc, param=PARAM):
    """
    simple check whether the electrodes keep responding.
    :param ele_data:
    :param frcc:
    :param param:
    :return:
    """
    trailing_type = frcc.trailing_type
    trailing_param = frcc.get_trailing_param(trailing_type)
    stats_test = param.stats_test

    frs = ele_data.firing_rates(frcc.trailing_type, trailing_param=trailing_param, fr_type=2)
    num_trials = len(frs)
    num_splits = int(num_trials / param.trial_per_test)
    splits = np.linspace(0, num_trials, num_splits, dtype=np.int)

    for i in range(1, len(splits)):
        tmp_responses = frs[splits[i-1] : splits[i]]
        if np.all(tmp_responses == 0):  # edge case, everything is zero
            return False
        avg_fr = robust_mean(tmp_responses)
        if stats_test == "binomial":
            num_positive = np.sum(tmp_responses > 0)
            total = np.sum(tmp_responses != 0) + 1
            p_value = scipy.stats.binom_test(num_positive, total, p=0.5)
            statistics = num_positive * 1.0 / total
        elif stats_test == "t_test":
            statistics, p_value = scipy.stats.ttest_1samp(tmp_responses, popmean=0)
        else:
            statistics, p_value = scipy.stats.wilcoxon(tmp_responses)
        if avg_fr < param.fr_lb or p_value > param.sig_level:
            return False
    return True


def stim_check(ele_data, frcc=frcc, param=PARAM):
    """
    simple check whether every stimuli is responding. Only include excitatory sites.
    :param ele_data:
    :param frcc:
    :param param:
    :return:
    """
    trailing_type = frcc.trailing_type
    trailing_param = frcc.get_trailing_param(trailing_type)
    stats_test = param.stats_test

    # initialize container
    _stim = []
    _p_value = []
    _avg_responses = []
    _statistics = []
    for stim in ele_data.stim:
        # print("The electrode and stimulus are %f and %f" % (electrode, stim))
        tmp_data = ele_data.get_kw_SpikeData(stim=stim)
        tmp_responses = tmp_data.firing_rates(trailing_type, trailing_param, fr_type=2)
        if len(tmp_responses) < 3:
            return False
        avg_fr = robust_mean(tmp_responses)
        if stats_test == "binomial":
            num_positive = np.sum(tmp_responses > 0)
            total = np.sum(tmp_responses != 0) + 1
            p_value = scipy.stats.binom_test(num_positive, total, p=0.5)
            statistics = num_positive * 1.0 / total
        elif stats_test == "t_test":
            statistics, p_value = scipy.stats.ttest_1samp(tmp_responses, popmean=0)
        else:
            statistics, p_value = scipy.stats.wilcoxon(tmp_responses)

        if avg_fr < param.fr_lb or p_value > param.sig_level:
            return False
    return True


def re_ext(directory, old_ext_pattern, new_ext):
    """
    Change the extension of files to new_ext
    intput:
        dir: directory where files are stored
        old_ext_pattern: a string, extension to be replaced in regular expression form, e.g.,
                         to change files with extension ".doc", old_ext_pattern should be "*.doc"
        new_ext: a string, new extension, e.g., new_ext: ".pdf"
    output:
        none
    """
    for path_name in glob.iglob(os.path.join(directory, old_ext_pattern)):
        name, ext = os.path.splitext(os.path.basename(path_name))
        os.rename(path_name, os.path.join(directory, name + new_ext))
    return (None)


def spike_times2spike_matrix(spike_times, dur=False, resolution=resolution, padding=0.005):
    """
        spike_times: a list of arrays, each contain spike times from one unit
            type: list of numpy arrays or lists
        dur: how long is the recording for these spike times
            default: False means unknown
        resolution: timing precision of the spike matrix, default is 0.1ms
        padding: zeros appended at the end for visualization, in seconds
    """
    if not dur:
        last_spike_timing = 0
        for timings in spike_times:
            if len(timings) == 0:
                continue
            max_time = timings.max()
            if max_time > last_spike_timing:
                last_spike_timing = max_time
    else:
        last_spike_timing = dur
    bins = np.arange(0, last_spike_timing + padding, resolution)
    spikematrix = []
    for idx, timings in enumerate(spike_times):
        spikematrix.append(np.histogram(timings, bins)[0])
    return np.asarray(spikematrix, dtype=np.bool)



# def node_edge_plot(node_dict, edge_list, edge_connectivities):
#     """
#     plot a graph with nodes at fixed locations and edges with certain connectivities
#     :param node_labels: node labels in a matrix
#     :param node_locations: a matrix of 2D locations (3 dimension numpy array) \n
#         the last dimension indicates the location, \n
#         the first two indicates the geometrical relationship (e.g., whether two locations are on the same row or columns)
#     :param edge_connectivities: strength of connectivities
#     :return:
#     """
#
#     def plot_curved_arrow(xs, ys, color, linewidth, alpha,
#                           st_idx=20, relative_arrow_size=15,
#                           head_width=0.75, head_length=0.5):
#         end_idx = -relative_arrow_size - st_idx
#         plt.plot(xs[st_idx:end_idx],
#                  ys[st_idx:end_idx], color=color,
#                  linewidth=linewidth, alpha=alpha)
#         arrow_sp = np.asarray([xs[end_idx], ys[end_idx]])
#         arrow_ep = np.asarray([xs[-relative_arrow_size], ys[-relative_arrow_size]])
#         arrow_delta = arrow_ep - arrow_sp
#         plt.arrow(arrow_sp[0], arrow_sp[1], arrow_delta[0], arrow_delta[1],
#                   fc=color, ec=color, linewidth=linewidth, alpha=alpha,
#                   head_width=head_width, head_length=head_length)
#
#     def find_path(edge, node_dict, resolution=1e-1, offset=0, distance=10):
#         straight_line_factor = 0.714 / 1.6
#         loc_list = [node_dict[node] for node in edge]
#         x_diff, y_diff = loc_list[1] - loc_list[0]
#
#         if np.max([np.abs(x_diff), np.abs(y_diff)]) <= distance:  # two electrodes are next to each other
#             ancher_xs = np.asarray([loc_list[0][0], loc_list[1][0]])
#             ancher_ys = np.asarray([loc_list[0][1], loc_list[1][1]])
#             if x_diff != 0:
#                 slope = (loc_list[1][1] - loc_list[0][1]) * 1.0 / (loc_list[1][0] - loc_list[0][0])
#                 intercept = loc_list[0][1] - slope * loc_list[0][0]
#                 xs = np.linspace(ancher_xs[0], ancher_xs[-1], np.int(np.abs(x_diff) / resolution))
#                 ys = xs * slope + intercept
#                 return (xs, ys + offset)
#             else:
#                 ys = np.linspace(ancher_ys[0], ancher_ys[-1], np.int(np.abs(y_diff) / resolution))
#                 xs = np.ones(ys.shape) * loc_list[0][0]
#                 return (xs, ys + offset)
#
#         elif x_diff == 0 and y_diff != 0:
#             y_avg = loc_list[0][1] + y_diff / 2.0
#             x_avg = loc_list[0][0] + np.sign(loc_list[0][0] - 15) * straight_line_factor * abs(y_diff) / 3.0
#             ancher_xs = np.asarray([loc_list[0][0], x_avg, loc_list[1][0]])
#             ancher_ys = np.asarray([loc_list[0][1], y_avg, loc_list[1][1]])
#             newys = np.linspace(ancher_ys[0], ancher_ys[-1], np.int(np.abs(y_diff) / resolution))
#             tmp_f = scipy.interpolate.interp1d(ancher_ys, ancher_xs, kind="quadratic")
#             newxs = tmp_f(newys)
#             return (newxs, newys + offset)
#
#         elif y_diff == 0 and x_diff != 0:
#             y_avg = loc_list[0][1] + np.sign(loc_list[0][1] - 15) * straight_line_factor * abs(x_diff) / 3.0
#             x_avg = loc_list[0][0] + x_diff / 2.0
#             ancher_xs = np.asarray([loc_list[0][0], x_avg, loc_list[1][0]])
#             ancher_ys = np.asarray([loc_list[0][1], y_avg, loc_list[1][1]])
#             newxs = np.linspace(ancher_xs[0], ancher_xs[-1], np.int(np.abs(x_diff) / resolution))
#             tmp_f = scipy.interpolate.interp1d(ancher_xs, ancher_ys, kind="quadratic")
#             newys = tmp_f(newxs)
#             return (newxs, newys + offset)
#
#         elif np.abs(y_diff * 1.0 / x_diff) == 1:
#             x_avg = np.mean([loc_list[0][0], loc_list[1][0]])
#             y_avg = np.mean([loc_list[0][1], loc_list[1][1]])
#             slope = y_diff * 1.0 / x_diff
#             x_avg, y_avg = np.asarray([x_avg, y_avg]) + \
#                            np.asarray([straight_line_factor / 2, -slope * straight_line_factor / 2]) * distance
#             ancher_xs = np.asarray([loc_list[0][0], x_avg, loc_list[1][0]])
#             ancher_ys = np.asarray([loc_list[0][1], y_avg, loc_list[1][1]])
#             newxs = np.linspace(ancher_xs[0], ancher_xs[-1], np.int(np.abs(x_diff) / resolution))
#             tmp_f = scipy.interpolate.interp1d(ancher_xs, ancher_ys, kind="quadratic")
#             newys = tmp_f(newxs)
#             return (newxs, newys + offset)
#
#         else:
#             slope = (loc_list[1][1] - loc_list[0][1]) * 1.0 / (loc_list[1][0] - loc_list[0][0])
#             intercept = loc_list[0][1] - slope * loc_list[0][0]
#             ancher_xs = np.asarray([loc_list[0][0], loc_list[1][0]])
#             xs = np.linspace(ancher_xs[0], ancher_xs[1], np.int(np.abs(x_diff) / resolution))
#             ys = xs * slope + intercept + offset
#             return (xs, ys + offset)
#
#     def create_bk(node_dict, figsize=(6, 6), fontsize=12, circle_size=2, distance=10):
#         fig = plt.figure(figsize=figsize)
#         ax = fig.gca()
#         x_list = []
#         y_list = []
#         for key in node_dict.keys():
#             node_label = key
#             tmp_x, tmp_y = node_dict[key]
#             x_list.append(tmp_x)
#             y_list.append(tmp_y)
#             plt.text(tmp_x, tmp_y, str(node_label),
#                      horizontalalignment='center', verticalalignment='center',
#                      fontsize=fontsize)
#             ax.add_artist(plt.Circle((tmp_x, tmp_y), circle_size, color='k', fill=False))
#         plt.xlim((np.min(x_list) - distance), (np.max(x_list) + distance))
#         plt.ylim((np.min(y_list) - distance), (np.max(y_list) + distance))
#         return (fig, ax)
#
#     create_bk(node_dict)
#     color_dict = {0: "blue", -1.0: "red", 1.0: "green"}
#     max_connectivity = max(edge_connectivities)
#     for idx, edge in enumerate(edge_list):
#         tmp_connectivity = edge_connectivities[idx] / max_connectivity
#         # tmp_certainty = edge_certainties[idx]
#         xs, ys = find_path(edge, node_dict, offset=0)
#         plot_curved_arrow(xs, ys, color_dict[np.sign(xs[-1] - xs[0])], tmp_connectivity, tmp_connectivity)



# archived functions
# class Subarray:
#     """
#     Define an object so that one could easily find the unique subarrays
#     in a 2D numpy array and use it for indexing
#     """
#
#     def __init__(self, nparray):
#         self.values = nparray
#         # unqarrays = np.vstack({tuple(row) for row in nparray})
#         npcopy = nparray.copy()
#         ncols = nparray.shape[1]
#         dtype = nparray.dtype.descr * ncols
#         struct = npcopy.view(dtype)
#         unq = np.unique(struct)
#         self.unique = unq
#         self.index = struct
#
#     def logidx(self, row_idx):
#         """
#         return the logix index of stimulus self.unique[row_idx]
#         :type row_idx: scalar
#         :return
#         """
#         stim_logidx = self.unique[row_idx] == self.index
#         return (stim_logidx)
#
#     def absidx(self, row_idx):
#         """
#         return the index of stimulus self.unique[row_idx]
#         :param row_idx:
#         :return:
#         """
#         stim_logidx = self.unique[row_idx] == self.index
#         stim_idx = np.flatnonzero(stim_logidx)
#         return stim_idx



