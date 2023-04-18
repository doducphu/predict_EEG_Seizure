import glob            # for file locations
import pandas as pd
import re
from IPython.display import display
import pprint          # for pretty printing
import sklearn
sklearn.__version__
pp = pprint.PrettyPrinter()
def file_list(folder_path, output=False):
    # create an empty list
    file_list = []
    # for file name in the folder path...
    for filename in glob.glob(folder_path):
        # ... append it to the list
        file_list.append(filename)

    # sort alphabetically
    file_list.sort()

    # Output
    if output:
        print(str(len(file_list)) + " files found")
        pp.pprint(file_list)

    return file_list

def data_load(file_path):
    # read in the datafile
    data = pd.read_csv(file_path,  # file in
                       header=None,  # no column names at top of file
                       dtype=float)  # read data as 'floating points' (e.g. 1.0)

    return data
def data_index(feat_data, file_name, output=False):
    # get the file identifier from the file (e.g. F001)
    file_identifier = re.findall('\w\d+', file_name)[0]
    # add this identifier to a column
    feat_data['file_id'] = file_identifier

    # if the file identifier has an S in...
    if re.findall('S', file_identifier):
        # make a class column with 'seizure' in
        feat_data['class'] = 'seizure'
    # ...otherwise...
    else:
        # .. make a class column with 'Baseline' in
        feat_data['class'] = 'baseline'

    # if the file identifier has a Z or O in...
    if re.findall('Z|O', file_identifier):
        # make a location column with 'surface' in
        feat_data['location'] = 'surface'
    # if the file identifier has an N in...
    elif re.findall('N', file_identifier):
        # make a location column with 'intracranial hippocampus' in
        feat_data['location'] = 'intracranial hippocampus'
    # if the file identifier has an S or F in...
    elif re.findall('S|F', file_identifier):
        # make a location column with 'intracranial epileptogenic zone' in
        feat_data['location'] = 'intracranial epileptogenic zone'

    # name the index
    feat_data.columns.name = 'feature'

    # add the file_id and class to the index
    feat_data = feat_data.set_index(['file_id', 'class', 'location'])
    # reorder the index so class is first, then file_id, then feature
    feat_data = feat_data.reorder_levels(['class', 'location', 'file_id'], axis='index')

    if output:
        display(feat_data)

    return feat_data
