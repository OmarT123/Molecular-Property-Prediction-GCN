import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import sys
sys.path.insert(0, './model')
from Graph2Property import Graph2Property
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def loadInputs(FLAGS, idx, modelName, unitLen): # Loads data from graph folders
    adj = None
    features = None
    adj = np.load('./database/'+FLAGS.database+'/adj/'+str(idx)+'.npy')
    features = np.load('./database/'+FLAGS.database+'/features/'+str(idx)+'.npy')
    # Graph adjacency list and features
    retInput = (adj, features)
    # Properties
    retOutput = (np.load('./database/'+FLAGS.database+'/'+FLAGS.output+'.npy')[idx*unitLen:(idx+1)*unitLen]).astype(float)

    return retInput, retOutput

def training(model, FLAGS, modelName):
    #Reading data from FLAGS
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    decay_rate = FLAGS.decay_rate
    save_every = FLAGS.save_every
    learning_rate = FLAGS.learning_rate
    num_DB = FLAGS.num_DB
    unitLen = FLAGS.unitLen

    total_iter = 0
    total_st = time.time()

    mse_total = []
    mae_total = []
    learning_rates = []

    mse = sys.float_info.max
    best_mse = sys.float_info.max
    best_model_dir = f'./save/{modelName}_best'
    if not os.path.exists(best_model_dir):
        os.mkdir(best_model_dir)

    print ("Start Training")
    for epoch in range(num_epochs):
        # Learning rate scheduling
        new_learning_rate = (learning_rate * (decay_rate ** epoch))
        model.assign_lr(new_learning_rate)
        learning_rates.append(new_learning_rate)

        for i in range(0,num_DB):
            _graph, _property = loadInputs(FLAGS, i, modelName, unitLen)
            num_batches = int(_graph[0].shape[0]/batch_size)

            st = time.time()
            for _iter in range(num_batches):
                total_iter += 1
                A_batch = _graph[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                X_batch = _graph[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                P_batch = _property[_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                #Train for 4 iterations then test on 1
                if total_iter % 5 != 0:
                    # Training
                    cost = model.train(A_batch, X_batch, P_batch)
                    print ("train_iter : ", total_iter, ", epoch : ", epoch, ", cost :  ", cost)

                elif total_iter % 5 == 0:
                    # Test accuracy
                    Y, cost = model.test(A_batch, X_batch, P_batch)
                    print ("test_iter : ", total_iter, ", epoch : ", epoch, ", cost :  ", cost)
                    # Calculate error every 100 iterations
                    if( total_iter % 100 == 0 ):
                        mse = (np.mean(np.power((Y.flatten() - P_batch),2)))
                        mae = (np.mean(np.abs(Y.flatten() - P_batch)))
                        train_summary_writer.add_summary(
                            tf.Summary(value=[tf.Summary.Value(tag='mse', simple_value=mse)])
                        )
                        train_summary_writer.add_summary(
                            tf.Summary(value=[tf.Summary.Value(tag='mae', simple_value=mae)])
                        )
                        mse_total.append(mse)
                        mae_total.append(mae)
                        print ("MSE : ", mse, "\t MAE : ", mae)

                # Save the parameters every 'save_every' iterations
                if total_iter % save_every == 0:
                    # Save network! 
                    ckpt_path = 'save/'+modelName+'.ckpt'
                    model.save(ckpt_path, total_iter)
                    if best_mse > mse:
                        model.save(best_model_dir+"/best_model.ckpt", epoch)
                        best_mse = mse

            et = time.time()
            print ("time : ", et-st)
            st = time.time()
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))
    mse_npy = np.array(mse_total)
    mae_npy = np.array(mae_total)
    return mse_npy, mae_npy



def plot_loss(gcn_train_loss, gcn_val_loss):
    """Plot the loss for each epoch

    Args:
        epochs (int): number of epochs
        train_loss (array): training losses for each epoch
        val_loss (array): validation losses for each epoch
    """
    plt.plot(gcn_train_loss, label="MSE")
    plt.plot(gcn_val_loss, label="MAE")
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Model Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def plot_targets(pred, ground_truth):
    """Plot true vs predicted value in a scatter plot

    Args:
        pred (array): predicted values
        ground_truth (array): ground truth values
    """
    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, ground_truth, s=0.5)
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    ax.axline((1, 1), slope=1)
    plt.xlabel("Predicted Value")
    plt.ylabel("Ground truth")
    plt.title("Ground truth vs prediction")
    plt.show()


# execution : ex)  python train.py GCN logP 3 100 0.001 0.95
# method = sys.argv[1] # GCN
# prop = sys.argv[2] # logP
# num_layer = int(sys.argv[3]) # 3
# epoch_size = int(sys.argv[4]) # 100 
# learning_rate = float(sys.argv[5]) # 0.001
# decay_rate = float(sys.argv[6]) # 0.95


database = 'QM9_deepchem'
numDB = 267
unit_len = 500
method = "GCN" # Can be set to GCN, GCN+a, GCN+g, GCN+a+g
prop = "mu" # Can be set to A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom

# Can experiment with different numbers for the following hyperparameters
num_layer = 3 
num_epochs = 150
learning_rate = 0.001
decay_rate = 0.95


# Get props

dbPath = "./database/QM9_deepchem/qm9.csv"
prop_path = f'./database/{database}/{prop}.npy'
if not os.path.exists(prop_path):
    df = pd.read_csv(dbPath)
    prop_list = df[prop].tolist()
    num_props = len(prop_list)
    # print(prop_list)
    print(num_props)

    np.save(prop_path, prop_list)



# database = ''
# if (prop in ['TPSA2', 'logP', 'SAS']):
#     database = 'QM9'
#     numDB = 13
#     unit_len = 10000
# elif (prop == 'pve'):
#     database = 'CEP'
#     numDB = 27
#     unit_len = 1000



print ('method :', method, '\t prop :', prop, '\t num_layer :', num_layer, '\t num_epochs :', num_epochs, '\t learning_rate :', learning_rate, '\t decay_rate :', decay_rate, '\t Database :', database, '\t num_DB :', numDB)


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', method, 'GCN, GCN+a, GCN+g, GCN+a+g') 
flags.DEFINE_string('output', prop, '')
flags.DEFINE_string('loss_type', 'MSE', 'Options : MSE, CrossEntropy, Hinge')  ### Using MSE  
flags.DEFINE_string('database', database, 'Options : ZINC, ZINC2')  ### Using MSEr 
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp') 
flags.DEFINE_string('readout', 'atomwise', 'Options : atomwise, graph_gather') 
flags.DEFINE_integer('latent_dim', 512, 'Dimension of a latent vector for autoencoder')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('epoch_size', num_epochs, 'Epoch size')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('save_every', 1000, 'Save every')
flags.DEFINE_float('learning_rate', learning_rate, 'Learning Rate')
flags.DEFINE_float('decay_rate', decay_rate, 'Decay Rate')
flags.DEFINE_integer('num_DB', numDB, '')
flags.DEFINE_integer('unitLen', unit_len, '')

modelName = FLAGS.model + '_' + str(FLAGS.num_layers) + '_' + FLAGS.output + '_' + FLAGS.readout + '_' + str(FLAGS.latent_dim) + '_' + FLAGS.database


log_dir = f"./logs/{modelName}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, 'train'))
    os.makedirs(os.path.join(log_dir, 'test'))

# Create summary writers for train and validation
train_summary_writer = tf.summary.FileWriter(log_dir+"/train")
# val_summary_writer = tf.summary.create_file_writer(log_dir + '/validation')



print ("Summary of this training & testing")
print ("Model name is", modelName)
print ("A Latent vector dimension is", str(FLAGS.latent_dim))
print ("Using readout funciton of", FLAGS.readout)
print ("A learning rate is", str(FLAGS.learning_rate), "with a decay rate", str(FLAGS.decay_rate))
print ("Using", FLAGS.loss_type, "for loss function in an optimization")

# _graph, _property = loadInputs(FLAGS, 0, 'modelName', unit_len)

# print(_graph[0].shape)
# print(_graph[0][0*FLAGS.batch_size:(0+1)*FLAGS.batch_size].shape)
# num_batches = int(_graph[0].shape[0]/FLAGS.batch_size)
# print(num_batches)
model = Graph2Property(FLAGS)
mse, mae = training(model, FLAGS, modelName)

train_summary_writer.close()


np.save(f"./mse_{prop}", mse)
np.save(f"./mae_{prop}", mae)

plot_loss(mse, mae)

#visualize learning_rates array