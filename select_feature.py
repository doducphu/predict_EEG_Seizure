import pandas as pd
import os
from IPython.display import display
from Seizure_Feature_Extraction import Seizure_Features
import sklearn
sklearn.__version__
from sklearn.preprocessing import LabelEncoder

from load_data import data_index,file_list,data_load

DOWNLOAD_DIR = "data"
EPIL_dir_file_list = file_list(os.path.join(DOWNLOAD_DIR, '*'), output=True)
feature_df = pd.DataFrame()
# iterate across the list of folders
for folder in EPIL_dir_file_list:
    # get a list of files in each folder
    folder_files_list = file_list(os.path.join(folder, '*'))
    # iteratate across the files in each folder
    for file in folder_files_list:
        # load the file
        df = data_load(file)
# display(df)
        feat = Seizure_Features(sf=173.61,
                                window_size=None,
                                feature_list=['power', 'power_ratio', 'mean',
                                              'mean_abs', 'std', 'ratio', 'LSWT'],
                                bandpasses=[[2, 4], [4, 8], [8, 12], [12, 30], [30, 70]])
        # transform the data using the function
        part_x_feat = feat.transform(df.values, channel_names_list=['CZ'])
        # put the numpy output back into a pandas df
        part_x_feat = pd.DataFrame(part_x_feat, columns=feat.feature_names)
        # re-index the data
        part_x_feat = data_index(part_x_feat, file)
        # if there is no data in the feature data so far...
        if feature_df.empty:
            # then make this the feature dataframe...
            feature_df = part_x_feat
        else:
            # ...otherwise combine the two dataframes together down the index
            feature_df = pd.concat([feature_df, part_x_feat], axis='index')

# display the dataframe

# reset the index into columns (for easy saving)
feature_df_save = feature_df.reset_index()
feature_df = feature_df_save

feature_df['location'].unique()
display(feature_df)
# # make a label encoder
le = LabelEncoder()
# # change the string labels to ints

# select only intracranial EEG
# feature_reduced = feature_df[feature_df.location != 'intracranial hippocampus']
# drop the columns which are not feature variables
feature_reduced_drop =feature_df.drop(['class', 'file_id', 'location'], axis='columns')

# change to an array
data_x = feature_reduced_drop.values
# change the string labels to ints
data_y = le.fit_transform(feature_df['class'])
