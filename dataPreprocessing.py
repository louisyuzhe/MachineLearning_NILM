import os
import random
import math
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape
# from keras.callbacks import ModelCheckpoint

# import numpy as np
# from numpy import array

from config import *

# appliance_name == aggregate for aggregate data
def get_filepath_for_appliance(appliance_name, house_name):
	label_path = DATA_SET_DIR + house_name + '/labels.dat'
	df = pd.read_csv(label_path, sep="\s+", header=None)
	
	list = df[df[1] == appliance_name][0].tolist()
	if list: 
		index = df[df[1] == appliance_name][0].tolist()[0]
		return DATA_SET_DIR + house_name + '/channel_' + str(index) + '.dat'
	else:
		return

def get_activation_for_appliance_house(appliance_name, house_name):
	appliance_path = get_filepath_for_appliance(appliance_name, house_name)
	
	dataset = {
		'start_time': [],
		'end_time': [],
		'duration': [],
		'house_name': [],
		'appliance_name': [],
	}
	
	if appliance_path:
		df_appliance = pd.read_csv(appliance_path, sep="\s+", header=None)
		
		''' get dataframe of activation '''
		threshold = APPLIANCE_CONFIG[appliance_name]['on_power_threshold']
		df_on = df_appliance[df_appliance[1] > threshold]
		min_on_duration = APPLIANCE_CONFIG[appliance_name]['min_on_duration']
		min_off_duration = APPLIANCE_CONFIG[appliance_name]['min_off_duration']
		
		# print(threshold, min_on_duration, min_off_duration)
		
		start_time = -1
		end_time = -1
		previous_index = -1
		
		# total_row = df_on.shape[0]
		# print(total_row)
		# count = 0
		for index, row in df_on.iterrows():
			time = row[0]
			power = row[1]
			
			# count += 1
			# if count % 10000 == 0:
				# print(appliance_name, house_name, count)
			
			if start_time == -1:
				start_time = time
				end_time = time
			else:
				if time - end_time <= min_off_duration:
					end_time = time
				elif index-previous_index == 1 and time - end_time <= GAP_FILLING_THRESHOLD:
					end_time = time
				else:
					# save
					if end_time - start_time >= min_on_duration:
						dataset['start_time'].append(start_time)
						dataset['end_time'].append(end_time)
						dataset['duration'].append(end_time-start_time)
						dataset['house_name'].append(house_name)
						dataset['appliance_name'].append(appliance_name)
					
					start_time = time
					end_time = time
			
			previous_index = index
		
		if end_time - start_time >= min_on_duration:
			dataset['start_time'].append(start_time)
			dataset['end_time'].append(end_time)
			dataset['duration'].append(end_time-start_time)
			dataset['house_name'].append(house_name)
			dataset['appliance_name'].append(appliance_name)
			
		
		# print(df_activation)
		''' ! get dataframe of activation '''
	else:
		print('error: no such appliance!')
		
		
	df_activation = pd.DataFrame(data=dataset)
	return df_activation

# generate and save activation for each appliance
def save_activation():
	list_df = []
	for key in APPLIANCE_CONFIG.keys():
		for house_name in HOUSE_NAME:
			print(key, house_name, df.shape)
			df = get_activation_for_appliance_house(key, house_name)
			list_df.append(df)
			
			# df.to_csv(PREPOCESSED_DATA_DIR + key + '_' + house_name + '.csv', sep=' ', index=False)
	result = pd.concat(list_df)
	result.to_csv(PREPOCESSED_DATA_DIR + 'activation.csv', sep=' ', index=False)

# save_activation()

# df_activation = pd.read_csv(PREPOCESSED_DATA_DIR + 'activation.csv', sep="\s+")
# print(df_activation)


def get_power_sequence(df, window_width, start_time):
	list_index = df[df[0].between(start_time, start_time+window_width, inclusive=True)].index.tolist()
	if len(list_index) == 0:
		return
	min_index = list_index[0]
	max_index = list_index[-1]
	
	index = min_index
	time = start_time
	previous_time = 0
	previous_power = 0
	current_time = df.at[index-1, 0] if index > 0  else 0
	current_power = df.at[index-1, 1] if index > 0  else 0
	sequence = []
	while index >= min_index and index <= max_index:
		previous_time = current_time
		previous_power = current_power
		current_time = df.at[index, 0]
		current_power = df.at[index, 1]
		
		if time >= start_time + window_width:
			break
		else:
			while time >= previous_time and time < current_time:
				if time - previous_time <= GAP_FILLING_THRESHOLD:
					sequence.append(previous_power)
				else:
					sequence.append(0)
				time += SAMPLE_WIDTH
		
			index += 1
	
	# zero padding
	seq_length = math.ceil(window_width / SAMPLE_WIDTH)
	current_length = len(sequence)
	sequence += [ 0 for i in range(seq_length-current_length)]
	
	return sequence

# path = get_filepath_for_appliance('aggregate', 'house_3')
# df = pd.read_csv(path, sep="\s+", header=None)
# list1 = get_power_sequence(df, APPLIANCE_CONFIG['kettle']['window_width'],  1363030535)
# path = get_filepath_for_appliance('kettle', 'house_3')
# df = pd.read_csv(path, sep="\s+", header=None)
# list2 = get_power_sequence(df, APPLIANCE_CONFIG['kettle']['window_width'],  1363030535)
# x = [ i for i in range(len(list1))]
# plt.plot(x, list1)
# plt.plot(x, list2)
# plt.show()

def generate_real_sample():
	

	df_activation = pd.read_csv(PREPOCESSED_DATA_DIR + 'activation.csv', sep="\s+")
	#for appliance_name in APPLIANCE_CONFIG.keys():
	#	if os.path.exists(PREPOCESSED_DATA_DIR + '/dataset_' + appliance_name + '.csv'):
	#		break

	if(1):
		print('loading dataframe')
		list_df_aggregate = {}
		for house_name in HOUSE_NAME:
			path = get_filepath_for_appliance('aggregate', house_name)
			df = pd.read_csv(path, sep="\s+", header=None)
			list_df_aggregate[house_name] = df
			
		list_df_appliance = {}
		for house_name in HOUSE_NAME:
			path = get_filepath_for_appliance(appliance_name, house_name)
			if path:
				df = pd.read_csv(path, sep="\s+", header=None)
				list_df_appliance[house_name] = df
		print('loading dataframe complete')
		
		# list of random choice house_3
		list_random_house = [ k for k in list_df_appliance.keys()]
		
		
		current_df_activation = df_activation[df_activation['appliance_name'] == appliance_name]
		window_width = APPLIANCE_CONFIG[appliance_name]['window_width']
		seq_length = math.ceil(window_width / SAMPLE_WIDTH)
		
		data_set = {}
		for i in range(seq_length):
			data_set['aggregate_power_' + str(i)] = []
			data_set['appliance_power_' + str(i)] = []
			data_set['timestamp_' + str(i)] = []
		data_set['house_name'] = []
		number_ignore = 0
		number_sample = math.floor((1 - SYNTHETIC_DATA_RATIO) * NUMBER_DATASET)
		for i in range(number_sample):
			if i % 1000 == 0:
				print( round(100 * i/number_sample, 1), '%')
			if random.random() < REAL_DATA_SAMPLEING_PROBABLITY:
				sample = current_df_activation.sample()
				start_time = sample['start_time'].tolist()[0]
				end_time = sample['end_time'].tolist()[0]
				shift_room = window_width - (end_time-start_time)
				shift = random.randint(0, shift_room)
				
				start_time += shift
				
				house_name = sample['house_name'].tolist()[0]
				
			else:
				house_name = random.choice(list_random_house)
				sample = list_df_appliance[house_name].sample()
				start_time = sample[0].tolist()[0]
			
			
			aggregate_seq = get_power_sequence(list_df_aggregate[house_name], window_width ,start_time)
			appliance_seq = get_power_sequence(list_df_appliance[house_name], window_width ,start_time)
			time_sequnce = [ start_time + SAMPLE_WIDTH * i for i in range(seq_length)]
			
			if aggregate_seq != None and appliance_seq != None:
				for i in range(seq_length):
					data_set['aggregate_power_' + str(i)].append(aggregate_seq[i])
					data_set['appliance_power_' + str(i)].append(appliance_seq[i])
					data_set['timestamp_' + str(i)].append(time_sequnce[i])
				
				data_set['house_name'].append(house_name)
			else:
				number_ignore += 1
			# plt.plot(time_sequnce, aggregate_seq)
			# plt.plot(time_sequnce, appliance_seq)
			# plt.show()
			
		df = pd.DataFrame(data_set)
		df.to_csv(PREPOCESSED_DATA_DIR + '/dataset_' + appliance_name + '.csv', sep=' ', index=False)
		print('ignore: ', number_ignore)
		
# generate_real_sample()

def generate_synthetic_sample():
	df_activation = pd.read_csv(PREPOCESSED_DATA_DIR + 'activation.csv', sep="\s+")
	print('loading dataframe')
	df_set = {}
	for appliance_name in APPLIANCE_CONFIG.keys():
		if os.path.exists(PREPOCESSED_DATA_DIR + '/synthetic_dataset_' + appliance_name + '.csv'):
			break
		df_set[appliance_name] = {}
		for house_name in HOUSE_NAME:
			path = get_filepath_for_appliance(appliance_name, house_name)
			if path:
				df_set[appliance_name][house_name] = pd.read_csv(path, sep="\s+", header=None)
	print('! loading dataframe')
	for appliance_name in APPLIANCE_CONFIG.keys():
		print(appliance_name)
		window_width = APPLIANCE_CONFIG[appliance_name]['window_width']
		seq_length = math.ceil(window_width / SAMPLE_WIDTH)
		
		data_set = {}
		for i in range(seq_length):
			data_set['aggregate_power_' + str(i)] = []
			data_set['appliance_power_' + str(i)] = []
			data_set['timestamp_' + str(i)] = []
		data_set['house_name'] = []
		
		
		number_sample = math.floor(SYNTHETIC_DATA_RATIO * NUMBER_DATASET)
		for i in range(number_sample):
			if i % 1000 == 0:
				print( round(100 * i/number_sample, 1), '%')
			list_noisy_seq = []
			for appliance_name_addon in APPLIANCE_CONFIG.keys():
				if appliance_name_addon != appliance_name and random.random() < SYNTHETIC_DATA_OTHER:
					sample = df_activation[df_activation['appliance_name'] == appliance_name_addon].sample()
					start_time = sample['start_time'].tolist()[0]
					start_time += random.randint(-window_width, window_width)
					house_name = sample['house_name'].tolist()[0]
					
					path = get_filepath_for_appliance(appliance_name_addon, house_name)
					if path:
						df = df_set[appliance_name_addon][house_name]
						appliance_seq = get_power_sequence(df, window_width ,start_time)
						list_noisy_seq.append(appliance_seq)
			
			if random.random() < SYNTHETIC_DATA_TARGET:
				sample = df_activation[df_activation['appliance_name'] == appliance_name].sample()
				start_time = sample['start_time'].tolist()[0]
				end_time = sample['end_time'].tolist()[0]
				shift_room = max(1, window_width - (end_time-start_time))
				shift = random.randint(0, shift_room)
				
				start_time += shift
				house_name = sample['house_name'].tolist()[0]
				
				path = get_filepath_for_appliance(appliance_name, house_name)
				if path:
					df = df_set[appliance_name][house_name]
					appliance_seq = get_power_sequence(df, window_width ,start_time)
					list_noisy_seq.append(appliance_seq)
			else:
				appliance_seq = np.array([0 for i in range(seq_length)])
					
			aggregate_seq = np.array([0 for i in range(seq_length)])
			for list in list_noisy_seq:
				aggregate_seq += np.array(list)
			
			time_sequnce = [ SAMPLE_WIDTH * i for i in range(seq_length)]
			
			for i in range(seq_length):
				data_set['aggregate_power_' + str(i)].append(aggregate_seq[i])
				data_set['appliance_power_' + str(i)].append(appliance_seq[i])
				data_set['timestamp_' + str(i)].append(time_sequnce[i])
			
			data_set['house_name'].append(house_name)
			
			# plt.plot(time_sequnce, aggregate_seq)
			# plt.plot(time_sequnce, appliance_seq)
			# plt.show()
			
		df = pd.DataFrame(data_set)
		df.to_csv(PREPOCESSED_DATA_DIR + '/synthetic_dataset_' + appliance_name + '.csv', sep=' ', index=False)	
	
# generate_synthetic_sample()

def standardize_dataset(appliance_name):
	df = pd.read_csv(PREPOCESSED_DATA_DIR + '/dataset_' + appliance_name + '.csv', sep="\s+")
	
	seq_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)
	
	# Standardisation
	print('standardize', appliance_name)
	# get std of random sample
	sample = df.sample(random_state=RANDOM_SEED)
	# sample = df.sample()
	aggregate_seq_sample = sample[[ 'aggregate_power_' + str(i) for i in range(seq_length)]]
	aggregate_seq_sample = np.array([aggregate_seq_sample['aggregate_power_' + str(i)].tolist()[0] for i in range(seq_length)])
	aggregate_seq_sample = aggregate_seq_sample - aggregate_seq_sample.mean()
	# print(aggregate_seq_sample, len(aggregate_seq_sample), np.std(aggregate_seq_sample))
	sample_std = np.std(aggregate_seq_sample)

	for i in range(seq_length):
		print(round(100* i/seq_length/2, 1), '%')
		i = str(i)
		new_column = pd.Series((df['aggregate_power_' + i] - df['aggregate_power_' + i].mean())/sample_std, name='aggregate_power_'+ i)
		df.update(new_column)
	
	max_power = APPLIANCE_CONFIG[appliance_name]['max_power']
	
	for i in range(seq_length):
		print(round(100* i/seq_length/2, 1), '%')
		i = str(i)
		new_column = pd.Series(df['appliance_power_' + i]/max_power, name='appliance_power_'+ i)
		df.update(new_column)
	
	df.to_csv(PREPOCESSED_DATA_DIR + '/standardized_dataset_' + appliance_name + '.csv', sep=' ', index=False)	

# for appliance_name in APPLIANCE_CONFIG.keys():
	# standardize_dataset(appliance_name)
# standardize_dataset('kettle')

def load_data(appliance_name, type='default'):
	df = pd.read_csv(PREPOCESSED_DATA_DIR + '/standardized_dataset_' + appliance_name + '.csv', sep="\s+")

	seq_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)
	df_input = df[[ 'aggregate_power_' + str(i) for i in range(seq_length)]]
	df_target = df[[ 'appliance_power_' + str(i) for i in range(seq_length)]]
	# print(df_target)
	X_train, X_test, y_train, y_test = train_test_split(df_input, df_target, test_size=1 / (1 + TRAIN_TEST_RATIO), random_state=RANDOM_SEED)

	print("start\n", X_train)
	return X_train, X_test, y_train, y_test


# a = np.array([[1,2,3,4]])

# print(np.std(a))


# print(os.path.exists(PREPOCESSED_DATA_DIR + 'activation1.csv'))

# print(np.array([1,2,3]) + np.array([1,2,3]))

# aggregate_path = get_filepath_for_appliance('aggregate', house_name)
# df_aggregate = pd.read_csv(appliance_path, sep="\s+", header=None)		
# window_width = APPLIANCE_CONFIG[appliance_name]['window_width']

# df_activation2 = get_activation_for_appliance_house('kettle', HOUSE_NAME[1])
# df_activation1 = get_activation_for_appliance_house('fridge', HOUSE_NAME[1])
# print(df_activation2)
# df_activation2 = get_activation_for_appliance_house('kettle', HOUSE_NAME[2])
# result = pd.concat([df_activation1, df_activation2])

# result.to_csv(PREPOCESSED_DATA_DIR + '/test.csv', sep=' ', index=False)
# print(APPLIANCE_CONFIG['kettle'])