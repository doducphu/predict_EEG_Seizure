import pickle

import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from IPython.display import display
from load_data import data_load
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
import sklearn
sklearn.__version__
from Seizure_Feature_Extraction import Seizure_Features

def runmodel(fileinput):
    feature_df1 = pd.DataFrame()
    df1 = data_load(fileinput)
    feat1 = Seizure_Features(sf=173.61,
                             window_size=None,
                             feature_list=['power', 'power_ratio', 'mean',
                                           'mean_abs', 'std', 'ratio', 'LSWT'],
                             bandpasses=[[2, 4], [4, 8], [8, 12], [12, 30], [30, 70]])
    # transform the data using the function
    part_x_feat1 = feat1.transform(df1.values, channel_names_list=['CZ'])
    # put the numpy output back into a pandas df
    part_x_feat1 = pd.DataFrame(part_x_feat1, columns=feat1.feature_names)
    # re-index the data

    if feature_df1.empty:
        # then make this the feature dataframe...
        feature_df1 = part_x_feat1
    else:
        # ...otherwise combine the two dataframes together down the index
        feature_df1 = pd.concat([feature_df1, part_x_feat1], axis='index')
    # display the dataframe
    display(feature_df1)
    class_names = ['baseline','seizure']  # fill the rest

    model = keras.models.load_model("modelCNN_LSTM.h5")


    # model.predict(fileinput)
    # model.compile(optimizer=Adam(0.001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', 'Recall', 'Precision'])
    #
    # classes = np.argmax(model.predict(feature_df1), axis=-1)
    # print(classes)
    #
    # names = [class_names[i] for i in classes]
    #
    # print(names)

runmodel("test1.txt")