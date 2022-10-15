import numpy as np
import pandas as pd
from scipy.signal import tukey, welch
from pywt import wavedec, swt
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from itertools import combinations
from numpy import inf
import math
from sklearn.preprocessing import StandardScaler
import os
def window(a, w, o, copy = False):
    # if there is no window to be applied
    if w == None:
        view = np.expand_dims(a.T, axis=0)
        
    # otherwise...
    else:
    
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        if o:
            view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
        else:
            view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::w]
    if copy:
        return view.copy()
    else:
        return view

def window_y(events, window_size, overlap, target=None, baseline=None):
    
  # window the data so each row is another epoch
  events_windowed = window(events, w = window_size, o = overlap, copy = True)
  
  if target:
    # turn to array of bools if seizure in the
    # windowed data
    bools = events_windowed == target
    # are there any seizure seconds in the data?
    data_y = np.any(bools,axis=1)
    # turn to 0's and 1's
    data_y = data_y.astype(int)
    # expand the dimensions so running down one column
    data_y = np.expand_dims(data_y, axis=1)
  
  elif baseline:
    # replace all baseline labels to nan
    data_y = pd.DataFrame(events_windowed).replace(baseline, np.nan)
    # get the most common other than baseline
    data_y = data_y.mode(1)
    # change nan back to baseline class
    data_y = data_y.fillna(baseline).values
    # if there was nothing but baseline there will be an empty array
    if data_y.size == 0:
        data_y = np.array([baseline]*data_y.shape[0])
        data_y = np.expand_dims(data_y, -1)
  
  else:
    # get the value most frequent in the window
    data_y = pd.DataFrame(events_windowed).mode(1).values

  return data_y
    
    
def bandpower(data, sf, weighted, mean, band):
    low, high = band
    
    # TODO: Not sure this does much...
    if weighted:
        weighted_window = ('tukey', 3)
        
    else:
        weighted_window = 'hann'

    # Compute the periodogram (Welch) furier method
    freqs, psd = welch(data,
                       sf,
                       window = weighted_window,
                       nperseg=(2 / low)*sf, # this ensures there are at least 2 cycles of the lowest frequency in the window
                       scaling='density',
                       axis=0
                      )
    
    # Find closest indices of band in frequency vector
    idx_min = np.argmax(np.round(freqs) > low) - 1
    idx_max = np.argmax(np.round(freqs) > high)
    
    #select frequencies of interest
    psd = psd[idx_min:idx_max,:]
    
    if mean:
        psd = np.nanmean(psd,0)
    else:
        psd = np.nanmedian(psd,0)
    
    return psd


def feature_append(all_features, data, axis=1, expand=True):
    if expand:
        data = np.expand_dims(data, axis=axis)
    
    # if the feature set is empty
    if all_features.size == 0:
        all_features = data
    else:
        all_features = np.concatenate((all_features, data), axis)
    
    return all_features


def pad_along_axis(array, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


def wavelet_decompose(data, feature_list, channel_name=None, wavelet='db4', wavelet_transform = 'DWT', level=6, scale=False):
    # bool to check if the data has been padded
    padded = False
    
    if wavelet_transform == 'DWT':
        # get the wavelet coefficients at each level in a list
        coeffs_list = wavedec(data, wavelet=wavelet, level=level)
    
    elif wavelet_transform == 'UDWT':
        # The signal length along the transformed axis be a multiple of 2**level
        atrous = (2**level)
        orig_shape = data.shape
        ceiled_len = math.ceil(orig_shape[1]/atrous)
        if orig_shape[1]/atrous != ceiled_len:
            padded_len = ceiled_len*atrous
            data = pad_along_axis(data, padded_len, axis=1)
            padded = True
        # get the wavelet coefficients at each level in a list
        coeffs_list = swt(data, wavelet=wavelet, level=level)
    
    if channel_name:
        # make an empty list for the features later
        feature_names = []
        
        # make a list of the component names
        nums = list(range(1,level+1))
        names=[]
        for num in nums:
            names.append('D' + str(num))
        # reverse the names so it counts down
        names = names[::-1]  
    
    # make empty arrays to help store data in later
    mean = np.array([])
    std = np.array([])
    LSWT = np.array([])
    mean_abs = np.array([])
    
    # for each decomposition level from the wavelets...
    for i, array in enumerate(coeffs_list):
        if wavelet_transform == 'DWT':
            # skip the first A
            if i == 0:
                continue
        elif wavelet_transform == 'UDWT':
            # just get the D's
            array = array[1]
            if padded:
                # remove the padding
                array = array[:orig_shape[0],:orig_shape[1]]
              
        if 'mean' in feature_list:
            # add the data straight into the wavelet_features array
            mean = feature_append(mean, np.mean(array,1))
        if 'std' in feature_list:
            # add the data straight into the wavelet_features array
            std = feature_append(std, np.std(array,1))
        if any(i in ['mean_abs','ratio'] for i in feature_list):
            # add the data into the mean_abs array
            mean_abs = feature_append(mean_abs, np.mean(np.absolute(array),1))
        if 'LSWT' in feature_list:
            # add the data into the LSWT array
            LSWT = feature_append(LSWT, np.sum(array,1))
    
    wavelet_features = np.array([])
    if 'mean' in feature_list:
        if scale:
            mean = frequency_scale(mean)
        # add the data straight into the wavelet_features array
        wavelet_features = feature_append(wavelet_features, mean, expand=False)
        # add the feature names
        if channel_name:
            feature_names.extend([channel_name+'|'+name+'_mean' for name in names])
    
    if 'std' in feature_list:
        if scale:
            std = frequency_scale(std)
        # add the data straight into the wavelet_features array
        wavelet_features = feature_append(wavelet_features, std, expand=False)
        # add the feature names
        if channel_name:
            feature_names.extend([channel_name+'|'+name+'_std' for name in names])
    
    if 'ratio' in feature_list:
        # make an empty df we will put data in
        ratio = np.empty((mean_abs.shape))
        # for each decomposition level
        for level in range(0, mean_abs.shape[1]):
            # for the first level
            if level == 0:
                ratio[:,level] = mean_abs[:,level]/mean_abs[:,level+1]
            # for the last level
            elif level == mean_abs.shape[1]-1:
                ratio[:,level] = mean_abs[:,level]/mean_abs[:,level-1]
            # all other levels
            else:
                mean_levels = (mean_abs[:,level-1]+mean_abs[:,level+1])/2
                ratio[:,level] = mean_abs[:,level]/mean_levels
        
        if scale:
            ratio = frequency_scale(ratio)
        # concat the ratio
        wavelet_features = feature_append(wavelet_features, ratio, expand=False)
        if channel_name:
            # add to the feature names
            feature_names.extend([channel_name+'|'+name+'_ratio' for name in names])
            
    if 'mean_abs' in feature_list:
        if scale:
            mean_abs = frequency_scale(mean_abs)
        # now add in the mean_abs to the feature list
        wavelet_features = feature_append(wavelet_features, mean_abs, expand=False)
        if channel_name:
            # add to the feature names
            feature_names.extend([channel_name+'|'+name+'_mean_abs' for name in names])
    
    if 'LSWT' in feature_list:        
        # minus the smallest value from each level for each time
        LSWT = LSWT.T - np.amin(LSWT,1)
        # transpose back
        LSWT = LSWT.T
        # plus 1 to each datapoint
        LSWT = LSWT+1
        # log each level for each time
        LSWT = np.log(LSWT)
        if scale:
            LSWT = frequency_scale(LSWT)
        # append the feature onto the wavelet features
        wavelet_features = feature_append(wavelet_features, LSWT, expand=False)
        if channel_name:
            # add to the feature names
            feature_names.extend([channel_name+'|'+name+'_LSWT' for name in names])
    
    if channel_name:
        return wavelet_features, feature_names
    
    else:
        return wavelet_features
    
    
def fft(time_data, fft_band):
    def replaceZeroes(data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data
    ab_fft = np.absolute(np.fft.rfft(time_data, axis=1)[:,fft_band[0]:fft_band[1]])
    ab_fft = replaceZeroes(ab_fft)
    return np.log10(ab_fft)


def correlation_matrix(data):
    # in the rare case that there is an inf
    # that came from the fft, turn it to a large number
    #data[data == -inf] = np.nan
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # scale data across each frequency
    scaled = scaler.fit_transform(data)
    # get the correlation coefficients from a Pearson product-moment (ignore nans)
    return np.ma.corrcoef(scaled)


# We get the eigenvalues and only take the first array out.
# We get the absolute to make them 'real'
def eigen(corr_matrix):
    # in the rare case that there is an inf or nan
    #corr_matrix = np.nan_to_num(corr_matrix)
    eigen_data = np.absolute(np.linalg.eig(corr_matrix)[0])
    # expand and transpose so it becomes columns
    eigen_data = np.expand_dims(eigen_data, axis=1).T
    
    return eigen_data

# essentially upper_right_triangle from MichaelHills
def corr_reshape(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        # the +1 excludes a channels correlations with itself
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])
            
    return np.expand_dims(np.array(accum), axis=0)

def entropy(data, feature_list, sf, channel_name=None):
    from entropy.entropy import sample_entropy, spectral_entropy
    entropy_features = np.array([])
    feature_names = []
    
    # change nans to 0's
    data = np.nan_to_num(data)
    
    if 'sample_entropy' in feature_list:
        sam_data = np.apply_along_axis(sample_entropy, 1, data)
        sam_data[sam_data == 0] = np.nan
        # now add in the data to the features
        entropy_features = feature_append(entropy_features, sam_data, expand=True)
        if channel_name:
            # add to the feature names
            feature_names.append(channel_name+'|sample_entropy')
    
    if 'spectral_entropy' in feature_list:
        spec_data = np.apply_along_axis(spectral_entropy, 1, data, sf,
                                        method='welch', nperseg = sf)
        # now add in the data to the features
        entropy_features = feature_append(entropy_features, spec_data, expand=True)
        if channel_name:
            # add to the feature names
            feature_names.append(channel_name+'|spec_entropy')
            
    if channel_name:
        return entropy_features, feature_names
    
    else:
        return entropy_features
    
    
def frequency_scale(data):
    SS = StandardScaler()
    orig_shape = data.shape
    # shape the data into one row
    data = data.reshape(-1, 1)
    # scale data in respect to all frequencies
    scaled_data = SS.fit_transform(data)
    # shape the data back to before
    scaled_data = scaled_data.reshape(orig_shape)
    
    return scaled_data
'''
=======================
CLASS: Seizure_Features
=======================

- sf
    - Sampling frequency
- downsample
    - Factor to downsample by
- window_size
    - Seconds(int)/datapoints(float) to epoch the data into
    - Can be None for no epoching
- overlap
    - Seconds(int)/datapoints(float) overlap between windows
    - Default None for no overlap
- weighted
    - If to apply a weighting to the window (default False)
- feature_list
    - list of features to be extracted
        - power:
        - power_ratio:
        - mean:
        - mean_abs:
        - std:
        - ratio:
        - LSWT:
        - fft_eigen:
        - fft_corr:
        - time_corr:
        - time_eigen:
        - sample_entropy: LIMITED IMPLIMENTATION
        - spectral_entropy: LIMITED IMPLIMENTATION
        - wavelet_coherence: NOT YET IMPLIMENTED
- bandpasses
    - list of bandpasses to extract for the power measure
- bandpass_mean
    - whether to take the mean or median of the Welch output
- bandpass_ratios
    - list of bandpasses to get a ratio between
- wavelet
    - type of wavelet to use
- wavelet_transform
    - type of transformation to use ('DWT' or 'UDWT')
- levels
    - how many levels to get from the wavelet transform
- fft_band
    - The fft band used for the fft_corr and fft_eigen methods
- scale
    - Whether to scale the data according to the mean so it has a standard deviation of 1.
    - Features based on frequency will be scaled in respect to each other.
    - If scikitlearn is >= 0.20.0 then you can leave NAN's in for the input if scaling
- target
    - the event target if doing binary classification
    - will override baseline if both provided
- baseline
    - the event target representing the class of least interest
    - if target and baseline both not provided then takes the most common class in window to classify window
'''

class Seizure_Features(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sf,
                 downsample=1,
                 window_size=1,
                 overlap=None,
                 weighted=False,
                 feature_list=['power', 'power_ratio', 'mean', 'mean_abs', 'std',
                               'ratio', 'LSWT', 'fft_corr', 'fft_eigen',
                                'time_corr', 'time_eigen', 'sample_entropy',
                               'spectral_entropy'],
                 bandpasses=[[1,4],[4,8]],
                 bandpass_mean=False,
                 bandpass_ratios=[[[3,12],[2,30]],],
                 wavelet = 'db4',
                 wavelet_transform = 'DWT',
                 levels=6,
                 fft_band=[1,48],
                 scale = False,
                 target=None, 
                 baseline=None
                ):
        self.sf = sf
        self.downsample = downsample
        if isinstance(window_size, int):
            self.window_size = window_size*sf
        else:
            self.window_size = window_size
        if isinstance(overlap, int):
            self.overlap = overlap*sf
        else:
            self.overlap = overlap
        self.weighted = weighted
        self.feature_list = feature_list
        self.bandpasses = bandpasses
        self.bandpass_ratios = bandpass_ratios
        self.bandpass_mean = bandpass_mean
        self.wavelet=wavelet
        self.wavelet_transform = wavelet_transform
        self.levels=levels
        self.fft_band = fft_band
        self.target = target
        self.baseline = baseline
        self.scale = scale
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, channel_names_list=None):
        # create empty arrays
        feature_set = np.array([])
        all_windowed_channels = np.array([])
        # create empty list
        feature_names = []
        # specify types of features
        wav_features = ['mean', 'mean_abs', 'std', 'ratio', 'LSWT']
        eigen_corr_features = ['fft_corr', 'fft_eigen', 'time_corr', 'time_eigen']
        entropy_features = ['app_entropy', 'spectral_entropy']
        
        # downsample
        X = X[::self.downsample]
        self.sf = self.sf/self.downsample
        
        # check if any of the rows have all the same number (maybe impedence testing?).
        # This will throw off most of the feature extraction so turn them to nans
        # Cant seem to think of a better way outside of pandas to do this!
        # Also we need to make sure there is more than one channel before doing this!
        if X.shape[1]>1:
          X = pd.DataFrame(X)
          all_same = X.eq(X.iloc[:, 0], axis=0).all(1)
          X.loc[all_same] = np.nan
          X = X.values
    
        # for each column of the data...
        for j, column in enumerate(X.T):
            # window the data so each row is another epoch
            channel_windowed = window(column, w = self.window_size, o = self.overlap, copy = True)

            # -----
            # POWER
            # -----
            if 'power' in self.feature_list:
                # create an empty array
                welch_data = np.array([])
                # for each bandpass in the bandpasses
                for bandpass in self.bandpasses:
                    # calculate the average or median of the frequency
                    # band for all time points in the channel
                    bandpass_data = bandpower(channel_windowed.T,
                                              sf = self.sf,
                                              weighted= self.weighted,
                                              mean = self.bandpass_mean,
                                              band = bandpass)
                    
                    welch_data = feature_append(welch_data, bandpass_data)

                    # if channel_names were provided...
                    if isinstance(channel_names_list, list):
                        # ...append the channel and frequency band to the list
                        feature_names.append(channel_names_list[j]+'|'+str(bandpass[0])+'_'+str(bandpass[1])+'Hz')
                
                if self.scale:
                    welch_data = frequency_scale(welch_data)

                # append the data straight into the feature set
                feature_set = feature_append(feature_set, welch_data, expand=False)
                

            # BANDPASS RATIOS
            if 'power_ratio' in self.feature_list:
                for bandpass_ratio in self.bandpass_ratios:
                    bandpass_1 = bandpower(channel_windowed.T,
                                              sf = self.sf,
                                              weighted= self.weighted,
                                              mean = self.bandpass_mean,
                                              band = bandpass_ratio[0])
                    bandpass_2 = bandpower(channel_windowed.T,
                                              sf = self.sf,
                                              weighted= self.weighted,
                                              mean = self.bandpass_mean,
                                              band = bandpass_ratio[1])
                    # divide bandpass 2 from 1
                    relative_power = bandpass_2/bandpass_1
                    
                    if self.scale:
                        relative_power = frequency_scale(relative_power)

                    # append the data straight into the feature set
                    feature_set = feature_append(feature_set, relative_power)
                    # if channel_names were provided...
                    if isinstance(channel_names_list, list):
                        ratio_str = str(bandpass_ratio[0][0])+'_'+str(bandpass_ratio[0][1])+'/'+str(bandpass_ratio[1][0])+'_'+str(bandpass_ratio[1][1])+'Hz'
                        # ...append the channel and frequency band to the list
                        feature_names.append(channel_names_list[j]+'|Ratio_'+ratio_str)
                        
                        
            # --------
            # WAVELETS
            # --------
            if any(i in wav_features for i in self.feature_list):
                # if channel_names were provided...
                if isinstance(channel_names_list, list):
                    # ... calculate all the requested wavelet features for the channel over
                    # all the epochs
                    wavelet_features, wavelet_feat_names = wavelet_decompose(channel_windowed,
                                                                             self.feature_list,
                                                                             channel_name=channel_names_list[j],
                                                                             wavelet=self.wavelet,
                                                                             wavelet_transform = self.wavelet_transform,
                                                                             level=self.levels,
                                                                             scale = self.scale)
                    # extend the feature list with the wavelet feature list
                    feature_names.extend(wavelet_feat_names)
                else:
                    # this is if we dont have the channel names
                    wavelet_features = wavelet_decompose(channel_windowed,
                                                         self.feature_list,
                                                         wavelet=self.wavelet,
                                                         wavelet_transform = self.wavelet_transform,
                                                         level=self.levels,
                                                         scale = self.scale)
                    
                # append the wavelet feature without expanding the data
                feature_set = feature_append(feature_set, wavelet_features, expand=False)
                
            # -------
            # Entropy
            # -------
            if 'sample_entropy' in self.feature_list or 'spectral_entropy' in self.feature_list:
                if isinstance(channel_names_list, list):
                    entropy_features, entropy_feat_names = entropy(channel_windowed,
                                                                   self.feature_list,
                                                                   self.sf,
                                                                   channel_name=channel_names_list[j])
                    # extend the feature list with the wavelet feature list
                    feature_names.extend(entropy_feat_names)
                else:
                    entropy_features = entropy(channel_windowed, self.feature_list, self.sf)
                    
                if self.scale:
                    SS = StandardScaler()
                    # scale data for each feature separately
                    entropy_features = SS.fit_transform(entropy_features)
                # append the wavelet feature without expanding the data
                feature_set = feature_append(feature_set, entropy_features, expand=False)
             
        # ----------
        # EIGEN CORR
        # ----------
            # if any of the correlation or eigenvalue methods have been specified...
            if any(i in eigen_corr_features for i in self.feature_list):
                # append the window data
                all_windowed_channels = feature_append(all_windowed_channels, channel_windowed, axis=2, expand=True)
        
        # if any of the correlation or eigenvalue methods have been specified...
        if any(i in eigen_corr_features for i in self.feature_list):
            # default bools so only need to check these rather than search
            # through a list each epoch which i assume would take longer?
            bool_dict = {'fft_eigen':False,
                         'fft_corr': False,
                         'time_eigen': False,
                         'time_corr': False}
            
            if 'fft_corr' in self.feature_list:
                bool_dict['fft_corr'] = True
            if 'fft_eigen' in self.feature_list:
                bool_dict['fft_eigen'] = True
            if 'time_corr' in self.feature_list:
                bool_dict['time_corr'] = True
            if 'time_eigen' in self.feature_list:
                bool_dict['time_eigen'] = True
            
            # create an empty array
            all_eigen_corr = np.array([])
            
            # go across epochs so we have channels and a single epoch
            # in the data
            for index, epoch in enumerate(all_windowed_channels):
                # create an empty array
                epoch_eigen_corr = np.array([])
                # for each key in the dictionary
                for key in bool_dict.keys():
                    # check it is activated
                    if bool_dict[key]:
                        # if there are any nans, inf
                        # in the data then we will
                        # just make this feature dataframe full of nan's so it doesnt crash
                        if np.isnan(epoch).any() or np.isinf(epoch).any():
                            if key in ['fft_corr','time_corr']:
                                # get the length of all possible channel combinations plus channels with themselves
                                len_combinations = len(list(combinations(range(epoch.shape[1]), 2)))
                                eigen_corr_data = np.full((1,len_combinations), np.nan)
                            else:    
                                eigen_corr_data = np.full((1,epoch.shape[1]), np.nan)
                        
                        # if there are no nans
                        else:
                            if key in ['fft_corr','fft_eigen']:
                                # get the fourier transform data
                                fft_data = fft(epoch.T, self.fft_band)
                                # get correlation matrix of channels over freq
                                corr_matrix = correlation_matrix(fft_data)
                            else:
                                # get correlation matrix of channel over time
                                corr_matrix = correlation_matrix(epoch.T)

                            # for the eigen data
                            if key in ['fft_eigen','time_eigen']:
                                # get absolute eigenvalues
                                eigen_corr_data = eigen(corr_matrix)
                            
                            # for the corr data
                            else:
                                eigen_corr_data = corr_reshape(corr_matrix)
                                        
                        # append the epoch feature without expanding the data
                        epoch_eigen_corr = feature_append(epoch_eigen_corr, eigen_corr_data, axis=1, expand=False)

                # append the feature without expanding the data
                all_eigen_corr = feature_append(all_eigen_corr, epoch_eigen_corr, axis=0, expand=False)
                        
            if self.scale:
                SS = StandardScaler()
                # scale data for each feature separately
                all_eigen_corr = SS.fit_transform(all_eigen_corr)
            # append all the eigen_corr data to the main feature set
            feature_set = feature_append(feature_set, all_eigen_corr, axis=1, expand=False)

            # if channel_names were provided...
            if isinstance(channel_names_list, list):
                # append the feature names
                if bool_dict['fft_eigen']:
                    feature_names.extend([channel+'|fft_eigen' for channel in channel_names_list])
                if bool_dict['time_eigen']:
                    feature_names.extend([channel+'|time_eigen' for channel in channel_names_list])
                if bool_dict['fft_corr'] or bool_dict['time_corr']:
                    # get all combinations of channels
                    combinations_list = list(combinations(channel_names_list, 2))
                    # join the channels together
                    corr_comb = ['_'.join(map(str,i)) for i in combinations_list]
                    if bool_dict['fft_corr']:
                        feature_names.extend([channel_comb+'|fft_corr' for channel_comb in corr_comb])
                    if bool_dict['time_corr']:
                        feature_names.extend([channel_comb+'|time_corr' for channel_comb in corr_comb])
                    
        # set feature names as a class attribute
        self.feature_names = feature_names
        
        # ------
        # DATA Y
        # ------
        # if y is an array
        if type(y).__module__ == np.__name__:
            y = y[::self.downsample]
            data_y = window_y(y, self.window_size, self.overlap, target=self.target, baseline=self.baseline)
            
            return feature_set, data_y
        else:
            return feature_set