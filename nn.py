import pandas 
import math

from keras.models import Sequential
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape
from keras.callbacks import ModelCheckpoint

import numpy as np
from numpy import array

import matplotlib.pyplot as plt

import random
import time
from res50_nt import Res50NTv1
from config import *
from dataPreprocessing import load_data
from helper import RNN_model, DAE_model, mean_abs_err, PTECA, get_agg_mean_std, unstandardize_aggregate_input
#from Data_Preprocessing import *

appliance_name = 'kettle'
BATCH_SIZE = 128
num_epochs = 20
# training / loading
mode = 'loading'

X_train, X_test, y_train, y_test = load_data(appliance_name)
sequence_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)
max_power = APPLIANCE_CONFIG[appliance_name]['max_power']

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

unstd_X_test = unstandardize_aggregate_input(X_test, appliance_name)


mode = 'training'

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
#y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

print('Training Data:')
print(X_train.shape)
print(y_train.shape)

print('Test Data:')
print(X_test.shape)
print(y_test.shape)

# trainning
# list_model = [RNN_model(sequence_length), DAE_model(sequence_length)]
# list_model = [DAE_model()]

model = Res50NTv1(input_shape=X_train.shape[1:])
model.summary()

if mode == 'training':
	start = time.time()
	checkpointer = ModelCheckpoint(filepath="model_" + appliance_name + '_1_' + str(num_epochs) + 'epo.hdf5', verbose=1, save_best_only=True)
	model.fit(X_train, y_train,
			  batch_size=BATCH_SIZE,
			  epochs=num_epochs,
			  validation_data=(X_test, y_test),
			  callbacks=[checkpointer])
	
	end = time.time()
	print('### Total trainning time cost: {} ###'. format(str(end - start)))
elif mode == 'loading':
	# !trainning
	print('loading from {}'.format("model_" + appliance_name + '_1_' + str(num_epochs) + 'epo.hdf5'))
	model.load_weights("model_" + appliance_name + '_1_' + str(num_epochs) + 'epo.hdf5')

	
	
######### improvement by status detection #########
def classify_on_off(input_matrix, max_power, threshold=0.5, off_window=60):
    result_matrix = np.zeros(input_matrix.shape)
    for i in range(result_matrix.shape[0]):
        current_window = 0
        start = False
        for j in range(result_matrix.shape[1]):
            if not start:
                if input_matrix[i,j] > max_power*threshold:
                    start = True
                    result_matrix[i,j] = 1
            else:
                if input_matrix[i,j] > max_power*threshold:
                    if current_window > 0:
                        for index in range(j-current_window ,j+1):
                            result_matrix[i, index] = 1
                        current_window = 0
                    else:
                        result_matrix[i,j] = 1
                else:
                    if current_window > off_window:
                        current_window = 0
                    else:
                        current_window += 1
    return result_matrix

def adjust_pre(status_matrix, pred):
    result = pred[:]
    for i in range(status_matrix.shape[0]):
        for j in range(status_matrix.shape[1]):
            if status_matrix[i,j] < 0.5:
                result[i,j,0] = 0
    return result
	
unstd_X_test = unstandardize_aggregate_input(X_test.reshape((X_test.shape[0], X_test.shape[1])), appliance_name)

status_matrix = classify_on_off(unstd_X_test, max_power)

preds = model.predict(X_test, verbose=0)

preds = adjust_pre(status_matrix, preds)
######### end of improvement by status detection #########
	
mae = mean_abs_err(preds, y_test) * max_power
print('MAE is {}'.format(mae))

# mae = mean_abs_err(mean_test, y_test) * max_power
# print('MAE is {}'.format(mae))

# pteca = PTECA(preds, y_test, X_test) 
# print('pteca is {}'.format(pteca))

'''
count  = 0
for index, list in enumerate(preds):
	plt.plot([ i for i in range(sequence_length)], [ item[0]*max_power  for item in list], label="Predict Value")
	plt.plot([ i for i in range(sequence_length)], [ item[0]*max_power  for item in y_test[index]], label="Ground Truth Value")
	# plt.plot([ i for i in range(sequence_length)], [ item[0]  for item in X_test[index]], label="Aggregate Value")
	plt.legend()
	# plt.legend({'Main',  'Appliance',  'Disaggregated'})
	plt.xlabel('Relative Time (6s)')
	plt.ylabel('Power (kw/h)')
	plt.show()
	count += 1
	if count  == 20:
		break
'''