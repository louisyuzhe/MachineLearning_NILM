import glob
import pandas as pd
import time
import numpy as np
import math
from config import *
from sklearn.model_selection import train_test_split

def join_multiple_channels():
    path = r'../dataset/'  # Path to dataset
    allFiles = glob.glob(path + "/*.dat")  # Data file name
    i = 0  # Counter for appliances array

    # Initialize the frame with timestamp (TS) as Index column
    frame1 = pd.DataFrame()
    # frame1.set_index('TS')

    # Loop to load dataset into a single frame
    for file_ in allFiles:
        start = time.time()
        # df = pd.read_csv(file_,delimiter = ' ' ,index_col=0, squeeze=True, header="None")
        df = pd.read_csv(file_, delimiter=' ', names=['TS', file_[34:43]], header=None)  # appliances[i]])
        print("Reading", i, " passed")
        frame1 = frame1.join(df.set_index('TS'), how='outer')
        print("Joining", i, " passed")

        end = time.time()
        print("Time used: ", end - start)

        i = i + 1

    # df.set_index('TS')
    print(frame1)

#Unimplemented
def load_appliance(appliance_name):
    path = 'dataset/'+appliance_name+ ".dat" # Path to dataset
    #file = glob.glob(path + "/"+ appliance_name+ ".dat")  # Data file name

    # Initialize the frame with timestamp (TS) as Index column
    #frameA = pd.DataFrame()
    # frame1.set_index('TS')

    # Load dataset into a frame
    start = time.time()

    df = pd.read_csv(path, delimiter=' ', names='TS', header=None)  # appliances[i]])
    print("Reading", " passed")

    end = time.time()
    print("Time used: ", end - start)

    #df.set_index('TS')
    print(df[1:10])

def combine_dataset(appliance_name, type='default'):
    if appliance_name == 'kettle':
        appliance_name = 'channel_5'
    path = 'dataset/'+appliance_name+ ".dat" # Path to dataset
    frame1 = pd.DataFrame()

    title = ['timestamp','appliance_power']
    df = pd.read_csv("dataset/channel_1.dat",sep = ' ', header = None, float_precision='round_trip', names=['timestamp','aggregate_power'])
    df2 = pd.read_csv(path, delimiter=' ', header=None, float_precision='round_trip', names=title)  # appliances[i]])

    frame1 = frame1.join(df.set_index('timestamp'), how='outer')
    frame1 = frame1.join(df2.set_index('timestamp'), how='outer')

    # Set the index into a frequency of 6 seconds
    end = int(frame1.index[len(frame1) - 1])
    start = int(frame1.index[0])
    print("start", start)
    print("end", end)

    l = [np.int64(i) for i in np.arange(start, end + 6, 6)]
    frame1 = frame1.reindex(l)

    # Forward filling
    frame1 = frame1.fillna(method='ffill')
    # Backward filling
    frame1 = frame1.fillna(method='bfill')
    frame1 = frame1.reset_index()
    columnsTitles = ["aggregate_power", "appliance_power", "timestamp"]
    frame1 = frame1.reindex(columns=columnsTitles)
    #print(frame1)
    return frame1

    #Slash data into 128 pieces
    #X_train, X_test, y_train, y_test = train_test_split(df_input, df_target, test_size=1 / (1 + TRAIN_TEST_RATIO), random_state=RANDOM_SEED)
    #return X_train, X_test, y_train, y_test

def save_activation(appliance_name):
    df = load_data(appliance_name)
    df['timestamp'].astype(np.int64)
    df.insert(3, "end_time", df['timestamp'], True)
    df.insert(4, "index_house", 1, True)
    df.insert(5, "name_appliance", 'oven', True)

    df.rename(columns={'timestamp':'start_time'}, inplace=True)
    columnsTitles = ["start_time", "end_time", "aggregate_power", "appliance_power", "index_house", "name_appliance"]
    df = df.reindex(columns=columnsTitles)

    #Disable warning
    pd.options.mode.chained_assignment = None  # default='warn'

    start = time.time()
    row_count = df.shape[0]
    for i in range (row_count):
        j =df['end_time'][i]
        df['end_time'][i] = j+6
    end = time.time()

    print("Time used: ", end - start)
    print(df)
    path2 = r'dataset/'  # Path to dataset
    df.to_csv(path2 + appliance_name + '_activation.csv', sep=",", index=False)

def generate_real_sample(appliance_name):
    path = r'dataset/tobe_std_'
    df = pd.read_csv(path+appliance_name+'.csv', delimiter=',')  # appliances[i]])
    #df = pd.read_csv(r'test.csv', delimiter=',')  # appliances[i]])

    seq_length = 128
    data_set = {}
    for i in range(seq_length):
        data_set['aggregate_power_' + str(i)] = []
        data_set['appliance_power_' + str(i)] = []
        data_set['timestamp_' + str(i)] = []
    data_set['house_name'] = []

    set=0
    total_set = int(len(df.index)/128)

    for j in range(total_set):
        for i in range(seq_length):
            data_set['aggregate_power_' + str(i)].append(df['aggregate_power'][i+set])
            data_set['appliance_power_' + str(i)].append(df['appliance_power'][i+set])
            data_set['timestamp_' + str(i)].append(df['timestamp'][i+set])
        data_set['house_name'].append(df['house_name'][i+set])
        set = set+128

    #print(data_set)
    df2 = pd.DataFrame(data_set)
    df2.to_csv('dataset/dataset_' + appliance_name + '.csv', sep=' ', index=False)


def standardize_dataset(appliance_name):
    df = pd.read_csv(r'dataset/dataset_'+ appliance_name + '.csv', sep="\s+")

    seq_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)

    # Standardisation
    print('standardize', appliance_name)
    # get std of random sample
    sample = df.sample()
    aggregate_seq_sample = sample[['aggregate_power_' + str(i) for i in range(seq_length)]]
    aggregate_seq_sample = np.array(
        [aggregate_seq_sample['aggregate_power_' + str(i)].tolist()[0] for i in range(seq_length)])
    aggregate_seq_sample = aggregate_seq_sample - aggregate_seq_sample.mean()
    # print(aggregate_seq_sample, len(aggregate_seq_sample), np.std(aggregate_seq_sample))
    sample_std = np.std(aggregate_seq_sample)

    for i in range(seq_length):
        print(round(100 * i / seq_length / 2, 1), '%')
        i = str(i)
        new_column = pd.Series((df['aggregate_power_' + i] - df['aggregate_power_' + i].mean()) / sample_std,
                               name='aggregate_power_' + i)
        df.update(new_column)

    max_power = APPLIANCE_CONFIG[appliance_name]['max_power']

    for i in range(seq_length):
        print(round(100 * i / seq_length / 2, 1), '%')
        i = str(i)
        new_column = pd.Series(df['appliance_power_' + i] / max_power, name='appliance_power_' + i)
        df.update(new_column)
    print(df)
    df.to_csv(r'dataset/standardized_dataset_' + appliance_name + '.csv', sep=' ', index=False)




def load_test_train_data(appliance_name, type='default'):
    df = pd.read_csv(PREPOCESSED_DATA_DIR + '/standardized_dataset_' + appliance_name + '.csv', sep="\s+")
    seq_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)
    df_input = df[[ 'aggregate_power_' + str(i) for i in range(seq_length)]]
    df_target = df[[ 'appliance_power_' + str(i) for i in range(seq_length)]]
    print(df_target)
    X_train, X_test, y_train, y_test = train_test_split(df_input, df_target, test_size=1 / (1 +
        TRAIN_TEST_RATIO), random_state=42)

    return X_train, X_test, y_train, y_test






