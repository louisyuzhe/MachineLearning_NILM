from keras.models import Sequential
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape

import numpy as np
import pandas as pd
from config import *
import math


def RNN_model(sequence_length):
	'''Creates the RNN module described in the paper
	'''
	model = Sequential()

	# 1D Conv
	model.add(Conv1D(16, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))

	#Bi-directional LSTMs
	model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
	model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

	# Fully Connected Layers
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
	# model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)

	return model

def DAE_model(sequence_length):
	'''Creates the Auto encoder module described in the paper
	'''
	model = Sequential()

	# 1D Conv
	model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))
	model.add(Flatten())

	# Fully Connected Layers
	model.add(Dropout(0.2))
	model.add(Dense((sequence_length-0)*8, activation='relu'))

	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))

	model.add(Dropout(0.2))
	model.add(Dense((sequence_length-0)*8, activation='relu'))

	model.add(Dropout(0.2))

	# 1D Conv
	model.add(Reshape(((sequence_length-0), 8)))
	model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	# model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)

	return model
	
def mean_abs_err(pred, ground_truth):
		# sum of error / number of sample
		sum_abs_error = 0
		for i in range(pred.shape[0]):
			for j in range(pred.shape[1]):
				sum_abs_error += abs(pred[i][j][0] - ground_truth[i][j][0])
				
		return sum_abs_error/(pred.shape[0] * pred.shape[1])

def PTECA(pred, ground_truth, aggre_data):
	sum_abs_error = 0
	sum_aggre = 0
	for i in range(pred.shape[0]):
		for j in range(pred.shape[1]):
			sum_abs_error += abs(pred[i][j][0] - ground_truth[i][j][0])
			sum_aggre += aggre_data[i][j][0]
			
	return 1 - sum_abs_error/ (2 * sum_aggre) 

def get_agg_mean_std(appliance_name):
    df = pd.read_csv(PREPOCESSED_DATA_DIR + '/dataset_' + appliance_name + '.csv', sep="\s+")

    seq_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)

    # get std of random sample
    sample = df.sample(random_state=RANDOM_SEED)
    aggregate_seq_sample = sample[[ 'aggregate_power_' + str(i) for i in range(seq_length)]]
    aggregate_seq_sample = np.array([aggregate_seq_sample['aggregate_power_' + str(i)].tolist()[0] for i in range(seq_length)])
    aggregate_seq_sample = aggregate_seq_sample - aggregate_seq_sample.mean()
    # print(aggregate_seq_sample, len(aggregate_seq_sample), np.std(aggregate_seq_sample))
    sample_std = np.std(aggregate_seq_sample)
    
    agg_mean = np.array([df['aggregate_power_' + str(i)].mean() for i in range(seq_length)])
    
    return agg_mean, sample_std
	
def unstandardize_aggregate_input(input_matrix, appliance_name):
    mean, std = get_agg_mean_std(appliance_name)
    unstd_matrix = input_matrix * std + mean
    
    return unstd_matrix