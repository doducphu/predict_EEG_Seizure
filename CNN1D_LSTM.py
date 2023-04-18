import pprint          # for pretty printing
from select_feature import data_x,data_y
# import pickle
import sklearn
sklearn.__version__
RANDOM_STATE = 0
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

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

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
TEST_SIZE = 0.15
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE)

# print('Training size: ' + str(X_train.shape))
# print('Validation size: ' + str(X_val.shape))
# print('Test size: ' + str(X_test.shape))
#
# print('Training size: ' + str(y_train.shape))
# print('Validation size: ' + str(y_val.shape))
# print('Test size: ' + str(y_test.shape))

from keras.layers import *
from keras.backend import clear_session
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np



X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)
print("X Train shape: ", X_train.shape)
print("X Test shape: ", X_test.shape)
print("X val shape: ",X_val.shape)
print("Y Train shape: ", y_train.shape)
print("Y test shape: ",y_test.shape)
print("Y val shape:",y_val.shape)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_val = np.reshape(X_val,(X_val.shape[0],X_val.shape[1],1))
# Returns a short sequential model
# def create_model(input_shape, flatten=False):
#     clear_session()
#     model = Sequential()
#
#     # this just tells the model what input shape to expect
#     model.add(Input(shape=input_shape))
#     for i in range(2):
#         model.add(Conv1D(filters=64,
#                          kernel_size=3,
#                          padding="same",
#                          activation='relu'))
#
#     model.add(MaxPooling1D(pool_size=3,  # size of the window
#                            strides=2,  # factor to downsample
#                            padding='same'))
#
#     for i in range(2):
#         model.add(Conv1D(filters=128,
#                          kernel_size=3,
#                          padding="same",
#                          activation='relu'))
#     # model.add(LSTM(100, return_sequences=True))
#     # model.add(Dropout(0.4))
#     if flatten:
#         model.add(Flatten())
#     else:
#         model.add(GlobalAveragePooling1D())
#
#
#
#     model.add(Dense(units=64,
#                     activation='relu'))
#
#     model.add(Dense(units=1,
#                     activation='sigmoid'))
#
#     model.compile(optimizer=Adam(0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', 'Recall', 'Precision'])
#     return model
#
# # Create a basic model instance
# model = create_model((X_train.shape[1],1))
# # # model = create_model(np.reshape(X_train, (X_train.shape[0], -1)))
# # # model= create_model( np.expand_dims(X_train,axis=1))
# model.summary()

def create_model(input_shape, flatten=False):
    clear_session()
    model = Sequential()

    # this just tells the model what input shape to expect
    model.add(Input(shape=input_shape))
    for i in range(2):
        model.add(Conv1D(filters=64,
                         kernel_size=3,
                         padding="same",
                         activation='relu'))

    model.add(MaxPooling1D(pool_size=3,
                           strides=2,
                           padding='same'))

    for i in range(2):
        model.add(Conv1D(filters=128,
                         kernel_size=3,
                         padding="same",
                         activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    if flatten:
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling1D())


    model.add(Dense(units=64,
                    activation='relu'))

    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Recall', 'Precision'])
    return model

# Create a basic model instance
model = create_model((X_train.shape[1],1))
# # model = create_model(np.reshape(X_train, (X_train.shape[0], -1)))
# # model= create_model( np.expand_dims(X_train,axis=1))
model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

def create_callbacks(best_model_filepath, tensorboard_logs_filepath):
    callback_checkpoint = ModelCheckpoint(filepath=best_model_filepath,
                                          monitor='val_loss',
                                          verbose=0,
                                          save_weights_only=False,
                                          save_best_only=False)

    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            verbose=1)

    callback_tensorboard = TensorBoard(log_dir=tensorboard_logs_filepath,
                                       histogram_freq=0,
                                       write_graph=False)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    return [callback_checkpoint, callback_early_stopping,
            callback_tensorboard, callback_reduce_lr]
from sklearn.utils.class_weight import compute_class_weight

EPOCHS = 200
BATCH_SIZE = 64
best_model_filepath = "CNN1D_Model.ckpt"
tensorboard_logs_filepath = "./CNN1D_logs/"

# calculate the class weights
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train),y=y_train)
class_weights = {i : class_weights[i] for i in range(2)}
# class_weights = dict(zip(np.unique(y_train), class_weights))
history_1D = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data = (X_val, y_val),
                       callbacks= create_callbacks(best_model_filepath,tensorboard_logs_filepath) ,class_weight = class_weights,verbose=1)
model.save('modelCNN_LSTM.h5')
# # Support Vector Machine
import matplotlib.pyplot as plt

def plot_progress(history_dict):
    for key in list(history_dict.keys())[:4]:
        plt.clf()  # Clears the figure
        training_values = history_dict[key]
        val_values = history_dict['val_' + key]
        epochs = range(1, len(training_values) + 1)

        plt.plot(epochs, training_values, 'bo', label='Training ' + key)

        plt.plot(epochs, val_values, 'b', label='Validation ' + key)

        if key != 'loss':
            plt.ylim([0., 1.1])

        plt.title('Training and Validation ' + key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.show()

plot_progress(history_1D.history)

# pipe_svc = Pipeline([('scl', StandardScaler()),
#                     ('clf', SVC(kernel='rbf',
#                                 class_weight = 'balanced',
#                                 probability=True,
#                                 random_state=RANDOM_STATE))])
# # classifier names
# classifier_names ='Super Vector Machine'
#
# # list of classifiers
# classifier = pipe_svc
#
# classifier.fit(X_train, y_train)
# # save model to disk
# pickle.dump(classifier,open('SVMClassifier.pkl','wb'))
# #load model to disk
# loaded_model = pickle.load(open('SVMClassifier.pkl','rb'))
# result = loaded_model.score(X_test,y_test)
# print(result)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Input, Reshape, Conv2D,MaxPooling2D, Flatten, Reshape, GlobalAveragePooling2D, SeparableConv2D, add
# from tensorflow.keras.backend import clear_session
# from tensorflow.keras.optimizers import Adam


# # Returns a short sequential model
# def create_model(data_shape):
#     model = Sequential()
#
#     model.add(Input(shape=data_shape[1:]))
#
#     for i in range(2):
#         model.add(Conv2D(filters=64,
#                          kernel_size=3,
#                          padding="same"))
#         model.add(BatchNormalization())
#         model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=2))
#
#     for i in range(2):
#         model.add(Conv2D(filters=128,
#                          kernel_size=3,
#                          padding="same"))
#         model.add(BatchNormalization())
#         model.add(Activation('relu'))
#
#     model.add(MaxPooling2D(pool_size=2))
#
#     model.add(GlobalAveragePooling2D())
#
#     model.add(Dense(units=100,
#                     activation='relu'))
#
#     model.add(Dense(units=50,
#                     activation='relu'))
#
#     model.add(Dense(units=1,
#                     activation='sigmoid'))
#
#     model.compile(optimizer=Adam(0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', 'AUC', 'Recall', 'Precision'])
#     return model

# clear_session()
# # Create a basic model instance
# model = create_model(X_train.shape)
# model.summary()









































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