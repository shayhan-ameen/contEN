__author__ = 'Shayhan'

folder_path= 'saved_variables'
output_figures_path= 'results'

CPU_USE = -2

SQ_LEN = 4320
D_0_TIME = int(SQ_LEN / 3)
D_1_TIME = int(SQ_LEN / 3) * 2


MAX_FEEDING = 3
ENS_INTERVAL = 4*60 # Search potential target BST after the following duration from the ENS given time


IC_WINDOW = 4*60 # 4 hours insulin and calories are taken for a patient to calculate insulin and calorie consumption factor of the patient within the duration
SEGMENT = 4
SEG_WINDOW = 2*60 # Duration of one segment
MAX_PREDICTION_DURATION = SEGMENT*SEG_WINDOW # Maximum duration for second day to predict





