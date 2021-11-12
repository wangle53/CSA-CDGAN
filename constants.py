"""
the dataset structure:
train
 |--train1
 |----t0.jpg, t1.jpg, label.jpg
 |--train2
 |----t0.jpg, t1.jpg, label.jpg
"""

# data path
DATASET = 'WHU-CD' # CDD WHU-CD LEVIR-CD

if DATASET == 'CDD':
    TRAIN_DATA = 'H:/RomoteSensingDataset/CDD/CDD'       # change to your dataset path
    TXT_PATH = './CDD/data'       # path to save .txt files
    TRAIN_TXT = './CDD/data/train.txt'        # training data
    TEST_TXT =  './CDD/data/test.txt'         # test data
    VAL_TXT = './CDD/data/validation.txt'      # validation data
    OUTPUTS_DIR = './CDD/outputs'                   # path to save training or test outputs
    IM_SAVE_DIR = './CDD/outputs/save_images'   # path to save training output images
    WEIGHTS_SAVE_DIR = './CDD/outputs/model'        # path to save training models
    BEST_WEIGHT_SAVE_DIR = './CDD/outputs/bestModel' # path to save the best performance model
    
elif DATASET == 'WHU-CD': 
    TRAIN_DATA = 'H:/RomoteSensingDataset/BCCD'       # change to your dataset path
    TXT_PATH = './WHU-CD/data'      
    TRAIN_TXT = './WHU-CD/data/train.txt'      
    TEST_TXT =  './WHU-CD/data/test.txt'         
    VAL_TXT = './WHU-CD/data/validation.txt'      
    OUTPUTS_DIR = './WHU-CD/outputs'                   
    IM_SAVE_DIR = './WHU-CD/outputs/save_images'   
    WEIGHTS_SAVE_DIR = './WHU-CD/outputs/model'     
    BEST_WEIGHT_SAVE_DIR = './WHU-CD/outputs/bestModel'
    
else:
    TRAIN_DATA = 'H:/RomoteSensingDataset/LEVIR-CD/dataset' # change to your dataset path
    TXT_PATH = './LEVIR-CD/data'      
    TRAIN_TXT = './LEVIR-CD/data/train.txt'      
    TEST_TXT =  './LEVIR-CD/data/test.txt'        
    VAL_TXT = './LEVIR-CD/data/validation.txt'     
    OUTPUTS_DIR = './LEVIR-CD/outputs'                  
    IM_SAVE_DIR = './LEVIR-CD/outputs/save_images'  
    WEIGHTS_SAVE_DIR = './LEVIR-CD/outputs/model'       
    BEST_WEIGHT_SAVE_DIR = './LEVIR-CD/outputs/bestModel'

    
# training configuration
EPOCH = 2000
ISIZE = 256         # input image size
BATCH_SIZE = 20
DISPLAY = True      # if display training phase in Visdom
DISPOLAY_STEP = 20    
RESUME = False       # if resume from the last epoch

# evaluation configuration
THRESHOLD = 0.5
SAVE_TEST_IAMGES = True # if save change maps during test

# optimizer configuration
LR = 0.0002     # learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 50
GAMMA = 0.1
G_WEIGHT = 200    # loss weight
D_WEIGHT = 1        # loss weight

# networks configuration
NC = 3      # input image channel size 
NZ = 100        # size of the latent z vector
NDF = 64        # the dimension size of the first convolutional of the generator
NGF = 64        # the dimension size of the first convolutional of the discriminator
EXTRALAYERS = 3 # add extral layers for the generator and discriminator
Z_SIZE = 16
GT_C = 1    # the channel size of ground truth
