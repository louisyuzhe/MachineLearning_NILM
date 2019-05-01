from Data_Preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt


#appliance_name = 'channel_3'
appliance_name = 'channel_3'
#X_train, X_test, y_train, y_test = load_data(appliance_name)
#load_data(appliance_name)
#standardize_dataset('oven')
#df= pd.read_csv(r'dataset/dataset_oven.csv', sep="\s+")
#print(df)

#save_activation(appliance_name)
#generate_dataset('oven')
df= pd.read_csv(r'prepocessed_data/standardized_dataset_kettle.csv', sep="\s+")
print(df)




