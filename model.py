import gdown
import os
import glob            # for file locations
import pprint          # for pretty printing
import numpy as np
import pickle
from select_feature import data_x,data_y
import pandas as pd
import seaborn as sns
import re
from IPython.display import display
from Seizure_Feature_Extraction import Seizure_Features
from load_data import data_load
import matplotlib.pyplot as plt
import sklearn
sklearn.__version__
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
RANDOM_STATE = 0
# # a = []
f = open("data\\F\\F001.txt", 'r')
# # for i in f:
# #     a.append(i.split())
# # print(a)

# # print(nums)
# # colours for printing outputs
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


pp = pprint.PrettyPrinter()


# def file_list(folder_path, output=False):
#     # create an empty list
#     file_list = []
#     # for file name in the folder path...
#     for filename in glob.glob(folder_path):
#         # ... append it to the list
#         file_list.append(filename)
#
#     # sort alphabetically
#     file_list.sort()
#
#     # Output
#     if output:
#         print(str(len(file_list)) + " files found")
#         pp.pprint(file_list)
#
#     return file_list
# # FILE_PATH = 'Epil_features.json.gzip'
# # Run the code and create/overwite old data?
# # EPIL_OVERWRITE = True
# #
# # # Where to save the feature data
# # EPIL_SAVE_PATH = 'C:\Users\ADMIN\Desktop\EEG\data'
# # if not os.path.exists(FILE_PATH):
# #     gdown.download('https://drive.google.com/uc?id=143dJfY1_U-p8elbvSzwe0TTHGcDo3PYv',
# #                 './'+FILE_PATH, quiet=False)
#
# #
# # print(color.BOLD + color.UNDERLINE + 'Title' + color.END)
# # print('Hello World')
# #
# # pp = pprint.PrettyPrinter()
# #
# #
# # def file_list(folder_path, output=False):
# #     # create an empty list
# #     file_list = []
# #     # for file name in the folder path...
# #     for filename in glob.glob(folder_path):
# #         # ... append it to the list
# #         file_list.append(filename)
# #
# #     # sort alphabetically
# #     file_list.sort()
# #
# #     # Output
# #     if output:
# #         print(str(len(file_list)) + " files found")
# #         pp.pprint(file_list)
# #
# #     return file_list
#
# def data_load(file_path, output=False):
#     # read in the datafile
#     data = pd.read_csv(file_path,  # file in
#                        header=None,  # no column names at top of file
#                        dtype=float)  # read data as 'floating points' (e.g. 1.0)
#
#     if output:
#         print(color.BOLD + color.UNDERLINE + '\n' + re.findall('\w\d+', file_path)[0] + color.END)
#         # Output detailed information on the data
#         print(color.BOLD + '\nData Information' + color.END)
#         data.info()
#
#         # Output first 5 rows and columns
#         print(color.BOLD + '\nDataframe Head' + color.END)
#         display(data.head())
#
#     return data
#
#
# def data_index(feat_data, file_name, output=False):
#     # get the file identifier from the file (e.g. F001)
#     file_identifier = re.findall('\w\d+', file_name)[0]
#     # add this identifier to a column
#     feat_data['file_id'] = file_identifier
#
#     # if the file identifier has an S in...
#     if re.findall('S', file_identifier):
#         # make a class column with 'seizure' in
#         feat_data['class'] = 'seizure'
#     # ...otherwise...
#     else:
#         # .. make a class column with 'Baseline' in
#         feat_data['class'] = 'baseline'
#
#     # if the file identifier has a Z or O in...
#     if re.findall('Z|O', file_identifier):
#         # make a location column with 'surface' in
#         feat_data['location'] = 'surface'
#     # if the file identifier has an N in...
#     elif re.findall('N', file_identifier):
#         # make a location column with 'intracranial hippocampus' in
#         feat_data['location'] = 'intracranial hippocampus'
#     # if the file identifier has an S or F in...
#     elif re.findall('F|S', file_identifier):
#         # make a location column with 'intracranial epileptogenic zone' in
#         feat_data['location'] = 'intracranial epileptogenic zone'
#
#     # name the index
#     feat_data.columns.name = 'feature'
#
#     # add the file_id and class to the index
#     feat_data = feat_data.set_index(['file_id', 'class', 'location'])
#     # reorder the index so class is first, then file_id, then feature
#     feat_data = feat_data.reorder_levels(['class', 'location', 'file_id'], axis='index')
#
#     if output:
#         display(feat_data)
#
#     return feat_data
#
# #
# # if EPIL_OVERWRITE:
# #     # delete the old version
# #     if os.path.exists(EPIL_SAVE_PATH):
# #         os.remove(EPIL_SAVE_PATH)
# DOWNLOAD_DIR = "data"
# if not os.path.exists(DOWNLOAD_DIR):
#     print("Error when loading data")
# else:
#     print("Already Downloaded")
#
# EPIL_dir_file_list = file_list(os.path.join(DOWNLOAD_DIR, '*'), output=True)
# feature_df = pd.DataFrame()
# # iterate across the list of folders
# for folder in EPIL_dir_file_list:
#     # get a list of files in each folder
#     folder_files_list = file_list(os.path.join(folder, '*'))
#     # iteratate across the files in each folder
#     for file in folder_files_list:
#         # load the file
#         df = data_load(file)
#
#         # setup the feature extraction function
#         feat = Seizure_Features(sf=173.61,
#                                 window_size=None,
#                                 feature_list=['power', 'power_ratio', 'mean',
#                                               'mean_abs', 'std', 'ratio', 'LSWT'],
#                                 bandpasses=[[2, 4], [4, 8], [8, 12], [12, 30], [30, 70]])
#         # transform the data using the function
#         part_x_feat = feat.transform(df.values, channel_names_list=['CZ'])
#         # put the numpy output back into a pandas df
#         part_x_feat = pd.DataFrame(part_x_feat, columns=feat.feature_names)
#         # re-index the data
#         part_x_feat = data_index(part_x_feat, file)
#         # if there is no data in the feature data so far...
#         if feature_df.empty:
#             # then make this the feature dataframe...
#             feature_df = part_x_feat
#         else:
#             # ...otherwise combine the two dataframes together down the index
#             feature_df = pd.concat([feature_df, part_x_feat], axis='index')
#
# # display the dataframe
#
# # reset the index into columns (for easy saving)
# feature_df_save = feature_df.reset_index()
# feature_df = feature_df_save
# display(feature_df)
# # reset the index into columns (for easy saving)
# # feature_df_save = feature_df.reset_index()
#
# # # save the dataframe to disk for later use
# # feature_df_save.to_json(EPIL_SAVE_PATH,
# #                         orient='index',
# #                         compression='gzip')
# # epil_baseline_file = os.path.join(EPIL_dir_file_list[0], 'F020.txt')
# # epil_seizure_file = os.path.join(EPIL_dir_file_list[3], 'S020.txt')
# #
# #
# #
# # def data_load(file_path, output=False):
# #     # read in the datafile
# #     data = pd.read_csv(file_path,header=None,dtype=float)
# #
# #     if output:
# #         print(color.BOLD + color.UNDERLINE + '\n' + re.findall('\w\d+', file_path)[0] + color.END)
# #         # Output detailed information on the data
# #         print(color.BOLD + '\nData Information' + color.END)
# #         data.info()
# #
# #         # Output first 5 rows and columns
# #         print(color.BOLD + '\nDataframe Head' + color.END)
# #         display(data.head())
# #
# #     return data
# # epil_baseline_df = data_load(epil_baseline_file, output=True)
# # epil_seizure_df = data_load(epil_seizure_file, output=True)
#
# # channel_name= ['CZ']
# # channel_type = ['eeg']
# # sample_rate = 173.61 # in hz
# #
# # # create an mne info file with meta data about the EEG
# # info = mne.create_info(ch_names=channel_name, sfreq=sample_rate,
# #                        ch_types=channel_type)
# #
# # # show the info file
# # display(info)
# #
# #
# # def mne_object(data, info, output=False):
# #     data = data.apply(lambda x: x * 1e-6)
# #     # transpose the data
# #     data_T = data.transpose()
# #     # create raw mne object
# #     raw = mne.io.RawArray(data_T, info)
# #
# #     return raw
# #
# #
# # epil_baseline_mne = mne_object(epil_baseline_df, info, output=True)
# # epil_seizure_mne = mne_object(epil_seizure_df, info, output=True)
# # plot_kwargs = {
# #     'scalings': dict(eeg=20e-4),   # zooms the plot out
# #     'highpass': 0.53,              # filters out low frequencies
# #     'lowpass': 40.,                # filters out high frequencies
# #     'n_channels': 1,               # just plot the one channel
# #     'duration': 24                # number of seconds to plot
# # }
# #
# # print(color.BOLD+color.UNDERLINE+"Inter-Ictal"+color.END)
# # epil_baseline_mne.plot(**plot_kwargs)
# # print(color.BOLD+color.UNDERLINE+"Ictal"+color.END)
# # epil_seizure_mne.plot(**plot_kwargs)
# #
# # for directory in EPIL_dir_file_list:
# #     # if re.findall('N|F|S',directory[-1]):
# #     # make a list of all the files in the directory
# #     files = file_list(os.path.join(directory, '*'))
# #     # randomly select 9 files from the list
# #     sampled_files = random.sample(files, 9)
# #
# #     fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
# #     x = 0
# #     y = 0
# #     for file in sampled_files:
# #
# #         # read in the datafile
# #         data = pd.read_csv(file,  # file in
# #                            header=None,  # no column names at top of file
# #                            dtype=float)  # read data as 'floating points' (e.g. 1.0)
# #
# #         # filter the data
# #         b, a = signal.butter(4, [1 / (sample_rate / 2), 30 / (sample_rate / 2)], 'bandpass', analog=False)
# #         filt_data = signal.filtfilt(b, a, data.T).T
# #
# #         axs[x, y].plot(filt_data)
# #         axs[x, y].set_title(re.findall('\w\d+', file)[0], pad=-15)
# #         # plot all of them on the same scale
# #         axs[x, y].set_ylim([-2100, 2100])
# #
# #         x += 1
# #
# #         if x == 3:
# #             y += 1
# #             x = 0
# #
# #     # add a big axes, hide frame
# #     fig.add_subplot(111, frameon=False)
# #     # hide tick and tick label of the big axes
# #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# #     plt.grid(False)
# #     plt.xlabel("Datapoints", labelpad=0.5)
# #     plt.ylabel("Microvolts (uV)", labelpad=20)
# #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
# #
# #     if directory[-1] == 'N':
# #         plt.title('Inter-ictal: Opposite Hippocampus')
# #
# #     elif directory[-1] == 'F':
# #         plt.title('Inter-ictal: Epileptogenic Zone')
# #
# #     elif directory[-1] == 'S':
# #         plt.title('Ictal: Epileptogenic Zone')
# #
# #     elif directory[-1] == 'Z':
# #         plt.title('Surface EEG: Eyes Open')
# #
# #     elif directory[-1] == 'O':
# #         plt.title('Surface EEG: Eyes Closed')
# #
# #     plt.show()
# # load features dataframe
# # feature_df = pd.read_json(FILE_PATH, orient='index', compression = 'gzip')
#
# # display examples of the data
# # display(feature_df.head())
# feature_df['location'].unique()
# # # select only intracranial EEG
# # feature_reduced = feature_df[feature_df.location != 'surface']
# # feature_reduced['location'].unique()
# #
# # feature_df_drop = feature_df.drop(['class', 'file_id', 'location'], axis='columns')
# # data_x = feature_df_drop.values
# #
# # display(feature_df['class'].value_counts())
# #
# # class_series = feature_df['class']
# #
# # # make a label encoder
# le = LabelEncoder()
# # # change the string labels to ints
# # data_y = le.fit_transform(class_series)
# # location_series = feature_df['location']
# # # get the unique labels
# # labels = list(class_series.unique())
# # # print out the labels and their new codes
# # for i, code in enumerate(list(le.transform(labels))):
# #     print(labels[i] + ': ' + str(code))
# #
# # one_hot_y = pd.get_dummies(location_series.unique())
# # dummy_y = pd.get_dummies(location_series.unique(), drop_first=True)
# #
# # print(color.BOLD+color.UNDERLINE+'Onehot'+color.END)
# # display(one_hot_y.head())
# # print(color.BOLD+color.UNDERLINE+'Dummy'+color.END)
# # display(dummy_y.head())
# # select only intracranial EEG
# feature_reduced = feature_df[feature_df.location != 'surface']
# # drop the columns which are not feature variables
# feature_reduced_drop = feature_reduced.drop(['class', 'file_id', 'location'], axis='columns')
# # change to an array
# data_x = feature_reduced_drop.values
# # change the string labels to ints
# data_y = le.fit_transform(feature_reduced['class'])

# print(color.BOLD+'Feature DataFrame'+color.END)
# display(data_x.shape)
# print(color.BOLD+'Target DataFrame'+color.END)
# display(data_y.shape)



# print(color.BOLD+color.UNDERLINE+'Feature DataFrame'+color.END)
# print('Training size: ' + str(X_train.shape))
# print('Validation size: ' + str(X_val.shape))
# print('Test size: ' + str(X_test.shape))
# print(color.BOLD+color.UNDERLINE+'\nTarget DataFrame'+color.END)
# print('Training size: ' + str(y_train.shape))
# print('Validation size: ' + str(y_val.shape))
# print('Test size: ' + str(y_test.shape))
#
#
# def get_proportions(data):
#     counts = pd.DataFrame(np.unique(data, return_counts=True), index=['Class_ID', 'Counts']).T
#     counts['Percent'] = (counts['Counts'] / counts['Counts'].sum()).round(2) * 100
#     counts = counts.set_index('Class_ID')
#     return counts
#
#
# print(color.BOLD + color.UNDERLINE + 'Training DataFrame' + color.END)
# display(get_proportions(y_train))
# print(color.BOLD + color.UNDERLINE + '\nTest DataFrame' + color.END)
# display(get_proportions(y_test))
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_train_scale = scaler.fit_transform(X_train)
#
# feature_list = list(feature_reduced_drop.columns)
# x_axis_label = 'CZ|D1_ratio'
# y_axis_label = 'CZ|D2_ratio'
#
# reduced_array = X_train_scale[:,[feature_list.index(x_axis_label),feature_list.index(y_axis_label)]]
# reduced_df = pd.DataFrame(reduced_array, columns=[x_axis_label, y_axis_label])
#
# display(reduced_df.head())
#
#
# sns.set(color_codes=True)
#
#
# def plot_pairplot(data_x, data_y):
#     data_plot = data_x.copy()
#     data_plot['class'] = np.vectorize({0: 'Baseline', 1: 'Seizure'}.get)(data_y)
#     sns.pairplot(data_plot,
#                  hue='class',
#                  hue_order=['Baseline', 'Seizure'],
#                  markers=["o", "s"],
#                  plot_kws=dict(alpha=0.5))
#
#     # plt.show(block=True)
#
# plot_pairplot(reduced_df, y_train)
#
#
# from sklearn.linear_model import LogisticRegression
#
# reg = LogisticRegression(C=100.,
#                          solver='liblinear',
#                          random_state=RANDOM_STATE)
#
# reg.fit(X_train_scale, y_train)
# vis_data = X_train_scale[:,[feature_list.index(x_axis_label),
#                           feature_list.index(y_axis_label)]]
# from mlxtend.plotting import plot_decision_regions
#
# reg.fit(vis_data, y_train)
#
# plot_decision_regions(vis_data,
#                       y_train,
#                       clf = reg)
#
# plt.xlabel(x_axis_label + ' [standardized]')
# plt.ylabel(y_axis_label + ' [standardized]')
# plt.show(block = True)
# from sklearn.pipeline import Pipeline
#
# pipe_reg = Pipeline([('scl', StandardScaler()),
#                      ('clf', LogisticRegression(C=0.00001,
#                                                 solver='liblinear',
#                                                 class_weight='balanced',
#                                                 random_state=RANDOM_STATE))])
#
#
# pipe_reg.fit(X_train, y_train)
# print('Validation Accuracy: %.3f' % pipe_reg.score(X_val, y_val))
# log_predicted = pipe_reg.predict(X_val)
# display(log_predicted)
# display(y_val)
#
# from sklearn.svm import SVC
#
# pipe_svc_linear = Pipeline([('scl', StandardScaler()),
#                             ('clf', SVC(C=100,
#                                         kernel='linear',
#                                         class_weight = 'balanced',
#                                         random_state=RANDOM_STATE))])
#
# from mlxtend.plotting import plot_decision_regions
#
# vis_data = X_train[:,[feature_list.index(x_axis_label),
#                       feature_list.index(y_axis_label)]]
#
# pipe_svc_linear.fit(vis_data, y_train)
#
# plot_decision_regions(vis_data,
#                       y_train,
#                       clf = pipe_svc_linear)
#
# plt.xlabel(x_axis_label)
# plt.ylabel(y_axis_label)
# plt.xlim(0,.6)
# plt.ylim(0,1.)
#
# plt.savefig('svm_linear_boundary.png')
# plt.show()
# pipe_svc_rbf = Pipeline([('scl', StandardScaler()),
#                          ('clf', SVC(C=100,
#                                      kernel='rbf',
#                                      class_weight = 'balanced',
#                                      random_state=RANDOM_STATE))])
#
# pipe_svc_rbf.fit(vis_data, y_train)
# pipe_svc_rbf.fit(X_train, y_train)
# print('Validation Accuracy: %.3f' % pipe_svc_rbf.score(X_val, y_val))
# from sklearn.neighbors import KNeighborsClassifier
# pipe_knn = Pipeline([('scl', StandardScaler()),
#                      ('clf', KNeighborsClassifier(n_neighbors=2))])
#
# pipe_knn.fit(vis_data, y_train)
#
# plot_decision_regions(vis_data,
#                       y_train,
#                       clf = pipe_knn)
#
# plt.xlabel(x_axis_label)
# plt.ylabel(y_axis_label)
# plt.xlim(0,.6)
# plt.ylim(0,1.)
# plt.show()
# pipe_knn.fit(X_train, y_train)
# print('Validation Accuracy: %.3f' % pipe_knn.score(X_val, y_val))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Seizure_Feature_Extraction import Seizure_Features
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.2
f_testmodel = open("data\\S\\S001.txt", 'r')
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)
feature_df1 = pd.DataFrame()
df1 = data_load(f_testmodel)
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
# reset the index into columns (for easy saving)

#
# feature_df1['location'].unique()
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# # Logistic Regression
# pipe_reg = Pipeline([('scl', StandardScaler()),
#                      ('clf', LogisticRegression(class_weight='balanced',
#                                                 solver = 'liblinear',
#                                                 random_state=RANDOM_STATE))])

# Support Vector Machine
pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(kernel='rbf',
                                class_weight = 'balanced',
                                probability=True,
                                random_state=RANDOM_STATE))])

# # Decision Tree
# DT = DecisionTreeClassifier(random_state=RANDOM_STATE)
#
# # K-Nearest Neighbours
# pipe_kkn = Pipeline([('scl', StandardScaler()),
#                     ('clf', KNeighborsClassifier())])

# classifier names
classifier_names ='Super Vector Machine'


# list of classifiers
classifier = pipe_svc

classifier.fit(X_train, y_train)
# save model to disk
pickle.dump(classifier,open('SVMClassifier.pkl','wb'))
#load model to disk
loaded_model = pickle.load(open('SVMClassifier.pkl','rb'))
result = loaded_model.score(X_test,y_test)
print(result)
test = loaded_model.predict(feature_df1)
if test == 0:
    test = "Baseline"
else:
    test = "Seizure"
print(test)

















































# from sklearn.metrics import confusion_matrix
#
#
# def pretty_confusion_matrix(confmat):
#     # this creates the matplotlib graph to make the confmat look nicer
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
#     for i in range(confmat.shape[0]):
#         for j in range(confmat.shape[1]):
#             ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
#
#     # change the labels
#     ax.set_xticklabels([''] + ['Baseline', 'Seizure'])
#     ax.set_yticklabels([''] + ['Baseline', 'Seizure'])
#
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     ax.xaxis.set_label_position('top')
#
#     plt.tight_layout()
#
#     plt.show()
#
#
# # use the first classifier to predict the validation set
# predictions = classifier.predict(X_val)
#
# # get the confusion matrix as a numpy array
# confmat = confusion_matrix(y_true=y_val, y_pred=predictions)
#
# # use the pretty function to make it nicer
# pretty_confusion_matrix(confmat)
# FP = confmat[0,1]
# TN = confmat[0,0]
# TP = confmat[1,1]
# FN = confmat[1,0]
#
# ERR = (FP+FN)/(FP+FN+TP+TN)
# ACC = 1-ERR
# TPR = TP/(FN+TP)
# FPR = FP/(FP+TN)
# PRE = TP/(TP+FP)
# REC = TP/(FN+TP)
# F1 = 2*((PRE*REC)/(PRE+REC))
#
# print('True positive rate (TPR): %.3f' % TPR)
# print('False positive rate (FPR): %.3f' % FPR)
# print('Error (ERR): %.3f' % ERR)
# print()
# print('Accuracy (ACC): %.3f' % ACC)
# print('Precision (PRE): %.3f' % PRE)
# print('Recall (REC): %.3f' % REC)
# print('F1-score (F1): %.3f' % F1)
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import accuracy_score
#
# print('Accuracy (ACC): %.3f' % accuracy_score(y_true=y_val, y_pred=predictions))
# print('Precision (PRE): %.3f' % precision_score(y_true=y_val, y_pred=predictions))
# print('Recall (REC): %.3f' % recall_score(y_true=y_val, y_pred=predictions))
# print('F1-score (F1): %.3f' % f1_score(y_true=y_val, y_pred=predictions))
# for i, classifier in enumerate(classifiers):
#   print(color.BOLD+color.UNDERLINE+classifier_names[i]+color.END)
#
#   predictions = classifier.predict(X_val)
#
#   confmat = confusion_matrix(y_true=y_val, y_pred=predictions)
#
#   pretty_confusion_matrix(confmat)
#   from sklearn.metrics import classification_report
#
#   for i, classifier in enumerate(classifiers):
#       predictions = classifier.predict(X_val)
#
#       classifier_score_df = pd.DataFrame
#           classification_report(y_val,
#                                 predictions,
#                                 target_names=['Baseline', 'Seizure'],
#                                 digits=2,
#                                 output_dict=True))
#
#       classifier_score_df.index.name = 'Metric'
#       classifier_score_df['Classifier'] = classifier_names[i]
#       classifier_score_df = classifier_score_df.set_index('Classifier', append=True)
#
#       if i == 0:
#           all_scores = classifier_score_df
#
#       else:
#           all_scores = pd.concat([all_scores, classifier_score_df])
#
#   display(all_scores.sort_index())
#   from mlxtend.plotting import plot_learning_curves
#
#   learning_curve_scoring = ['accuracy', 'precision']
#
#   print(color.BOLD + color.UNDERLINE + classifier_names[1] + color.END)
#   for scoring_method in learning_curve_scoring:
#       plot_learning_curves(X_train, y_train, X_val, y_val, classifiers[1],
#                            train_marker='o', test_marker='^',
#                            scoring=scoring_method, print_model=False)
#       if scoring_method in ['accuracy', 'precision']:
#           plt.ylim(top=1.01)
#       plt.show()

# import numpy as np
# from sklearn.model_selection import RepeatedKFold
# from sklearn.base import clone
#
# N_SPLITS = 2
# N_REPEATS = 2
#
# RepKFold = RepeatedKFold(n_splits=N_SPLITS,
#                          n_repeats=N_REPEATS,
#                          random_state=RANDOM_STATE)
#
# scores = []
# i = 0
# for train_index, validation_index in RepKFold.split(X_train, y_train):
#     clone_clf = clone(pipe_reg)
#
#     clone_clf.fit(X_train[train_index], y_train[train_index])
#     score = clone_clf.score(X_train[validation_index], y_train[validation_index])
#     scores.append(score)
#
#     print(color.BOLD + color.UNDERLINE + 'Fold ' + str(i + 1) + color.END)
#     print('Class dist.: %s, Acc: %.3f' % (np.bincount(y_train[train_index]), score))
#     print("\nTRAIN:", train_index)
#     print("\nVALIDATION:", validation_index)
#     print()
#     i += 1
#
# print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
#
#
# from scipy.stats import expon
#
# plt.hist(expon(scale=0.1).rvs(size=1000), histtype='stepfilled', alpha=0.2)
# plt.xlabel('Parameter Number')
# plt.ylabel('Times picked')
# plt.show()
# from scipy.stats import gamma
#
# plt.hist(gamma(a=10, scale=0.01).rvs(size=1000), histtype='stepfilled', alpha=0.2)
# plt.xlabel('Parameter Number')
# plt.ylabel('Times picked')
# plt.show()
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# PARAM_DIST = {
#     'C': hp.uniform('C', 0, 8),
#     'kernel': hp.choice('kernel', [
#         {'ktype': 'linear', 'gamma': 'auto'},  # gamma ignored
#         {'ktype': 'sigmoid', 'gamma': hp.uniform('sig_gamma', 0, 1)},
#         {'ktype': 'poly', 'gamma': hp.uniform('poly_gamma', 0, 1)},
#         {'ktype': 'rbf', 'gamma': hp.uniform('rbf_gamma', 0, 1)}]),
#     'scale': hp.choice('scale', [0, 1])
# }
#
#
# def hyperopt_train_test(params):
#     X_ = X_train[:]
#
#     if 'scale' in params:
#         if params['scale'] == 1:
#             sc = StandardScaler()
#             X_ = sc.fit_transform(X_)
#
#     clf = SVC(C=params['C'],
#               kernel=params['kernel']['ktype'],
#               gamma=params['kernel']['gamma'],
#               )
#
#     return cross_val_score(clf, X_, y_train, cv=5).mean()
#
#
# def objective(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc,  # minus because we need to reduce
#             'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(objective, PARAM_DIST,
#             algo=tpe.suggest, max_evals=500,
#             trials=trials)
#
# print('best:')
# print(best)
# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# from hyperopt import tpe
# # from hyperopt import hp
# from hpsklearn import HyperoptEstimator, svc
# from hpsklearn import standard_scaler, min_max_scaler, normalizer
#
#
#
# X_train, X_test, y_train, y_test = train_test_split(data_x,
#                                                     data_y,
#                                                     test_size=TEST_SIZE,
#                                                     random_state=RANDOM_STATE)
# if __name__ == '__main__':
#     estim = HyperoptEstimator(classifier= svc("my_clf"),
#                               preprocessing=standard_scaler("my_pre"),
#                               algo=tpe.suggest,
#                               max_evals=500,
#                               trial_timeout=120)
#
#     # Search the hyperparameter space based on the data
#     estim.fit(X_train, y_train)
#
#     # Show the results
#     print(estim.score(X_test, y_test))
#     # 1.0
#
# print(estim.best_model())
# estim = HyperoptEstimator(
#     classifier=svc('mySVC',
#                    kernels=['linear', 'rbf', 'poly', 'sigmoid'],
#                    C = hp.uniform('C', 0, 8),
#                    gamma = hp.uniform('gamma', 0, 1)),
#     preprocessing = [standard_scaler('standard'),
#                      min_max_scaler('minmax'),
#                      normalizer('Normal')],
#     max_evals=300
# )
#
# estim.fit(X_train,y_train)
#
# predictions = estim.predict(X_test)
#
# confmat = confusion_matrix(y_true=y_test, y_pred=predictions)
#
# pretty_confusion_matrix(confmat)