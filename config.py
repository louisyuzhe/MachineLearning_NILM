# data within threshold min do forward-filling, otherwise 0 filling. Unit: second
GAP_FILLING_THRESHOLD = 180

# sample every 6s
SAMPLE_WIDTH = 6

# ['kettle', 'fridge', 'washing_machine', 'microwave', 'dishwasher', 'oven']
APPLIANCE_CONFIG = {
    'kettle': {
        # count in second
        'window_width': 128 * SAMPLE_WIDTH,
        'max_power': 3100,
        'on_power_threshold': 2000,
        'min_on_duration': 12,
        'min_off_duration': 0
    },

    'fridge': {
        # ???
        'window_width': 512 * SAMPLE_WIDTH,
        'max_power': 300,
        'on_power_threshold': 50,
        'min_on_duration': 60,
        'min_off_duration': 12
    },
    'washing_machine': {
        # ???
        'window_width': 1334 * SAMPLE_WIDTH,
        'max_power': 2500,
        'on_power_threshold': 20,
        'min_on_duration': 1800,
        'min_off_duration': 160
    },
    'microwave': {
        # ???
        'window_width': 310 * SAMPLE_WIDTH,
        'max_power': 3000,
        'on_power_threshold': 200,
        'min_on_duration': 12,
        'min_off_duration': 30
    },
    'dishwasher': {
        'window_width': 1536 * SAMPLE_WIDTH,
        'max_power': 2500,
        'on_power_threshold': 10,
        'min_on_duration': 1800,
        'min_off_duration': 1800
    },
    'oven': {
        'window_width': 1536 * SAMPLE_WIDTH,
        'max_power': 1725,
        'on_power_threshold': 600,
        'min_on_duration': 12,
        'min_off_duration': 30
    }

}

# probability that sample includes target applicance's activation
REAL_DATA_SAMPLEING_PROBABLITY = 0.5

# the ratio of synthetic data to the whole dataset
SYNTHETIC_DATA_RATIO = 0.5
# probablity that includes the target applicance
SYNTHETIC_DATA_TARGET = 0.5
# probablity that includes the other applicance
SYNTHETIC_DATA_OTHER = 0.25

DATA_SET_DIR = 'ukdale_mini/'
HOUSE_NAME = ['house_1', 'house_2', 'house_3', 'house_4', 'house_5']

PREPOCESSED_DATA_DIR = 'prepocessed_data/'

NUMBER_DATASET = 20000
TRAIN_TEST_RATIO = 9
TRAIN_VAL_RATIO = 9

RANDOM_SEED = 123