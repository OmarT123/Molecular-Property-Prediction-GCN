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
from plotter import plot_loss
import time
import datetime
import math

# DONT NEED
def loadInputs(FLAGS, idx, model_name, unit_len): # Loads data from graph folders
    adj = None
    features = None
    adj = np.load('./database/'+FLAGS.database+'/adj/'+str(idx)+'.npy')
    features = np.load('./database/'+FLAGS.database+'/features/'+str(idx)+'.npy')
    # Graph adjacency list and features
    retInput = (adj, features)
    # Properties
    retOutput = (np.load('./database/'+FLAGS.database+'/'+FLAGS.output+'.npy')[idx*unit_len:(idx+1)*unit_len]).astype(float)

    return retInput, retOutput

# DONT NEED
def loadData(path, sub_path, idx, unit_len, prop_name):
    all_adj = os.listdir(os.path.join(path, sub_path,'adj'))

    file_name = all_adj[idx]
    file_idx = int(file_name.split('.')[0])

    adj = np.load(os.path.join(path, sub_path, 'adj', file_name))
    features = np.load(os.path.join(path, sub_path, 'features', file_name))
    graph = (adj, features)

    prop = (np.load(os.path.join(path, prop_name+".npy"))[file_idx*unit_len:(file_idx+1)*unit_len]).astype(float)

    return graph, prop

database = 'QM9_deepchem'
numDB = 267
unit_len = 500
method = "GCN+a+g" # Can be set to GCN, GCN+a, GCN+g, GCN+a+g
prop = "mu" # Can be set to A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom

# Can experiment with different numbers for the following hyperparameters
num_layer = 3
num_epochs = 100
learning_rate = 0.001
decay_rate = 0.95

# DONT NEED
dbPath = "./database/QM9_deepchem/qm9.csv"
prop_path = f'./database/{database}/{prop}.npy'
if not os.path.exists(prop_path):
    df = pd.read_csv(dbPath)
    prop_list = df[prop].tolist()
    num_props = len(prop_list)
    print(num_props)

    np.save(prop_path, prop_list)

print ('method :', method, '\t prop :', prop, '\t num_layer :', num_layer, '\t num_epochs :', num_epochs, '\t learning_rate :', learning_rate, '\t decay_rate :', decay_rate, '\t Database :', database, '\t num_DB :', numDB)

# Unnecessary, only used to read summary
time.sleep(5)


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

model_name = FLAGS.model + '_' + str(FLAGS.num_layers) + '_' + FLAGS.output + '_' + FLAGS.readout + '_' + str(FLAGS.latent_dim) + '_' + FLAGS.database

print ("Summary of this training & testing")
print ("Model name is", model_name)
print ("A Latent vector dimension is", str(FLAGS.latent_dim))
print ("Using readout function of", FLAGS.readout)
print ("A learning rate is", str(FLAGS.learning_rate), "with a decay rate", str(FLAGS.decay_rate))
print ("Using", FLAGS.loss_type, "for loss function in an optimization")

# Unnecessary, only used to read summary
time.sleep(5)

# Initialize your model
model = Graph2Property(FLAGS)



# Define hyperparameters
num_epochs = FLAGS.epoch_size
batch_size = FLAGS.batch_size

# DONT NEED
# adj_df = np.load('./database/QM9_deepchem/new_adj.npy')
# feat_df = np.load('./database/QM9_deepchem/new_features.npy')
# properties = pd.read_csv('./database/QM9_deepchem/qm9.csv')[prop]
# graphs = (adj_df, feat_df)

#DONT NEED
training_range = [0, 213] # 214 * 500 = 107_000 molecules
validation_range = [214, 240] # 27 * 500 = 13_500 molecules
test_range = [241, 267] # 27 * 500 = 13_500 molecules (actually 13_385 exactly)

# DONT NEED
# training_idx = len(os.listdir('./database/QM9_deepchem/training_dataset/adj'))
# validation_idx = len(os.listdir('./database/QM9_deepchem/validation_dataset/adj'))
# testing_idx = len(os.listdir('./database/QM9_deepchem/testing_dataset/adj'))

# Graphs
training_graphs = (np.load('./database/QM9_deepchem/train_adj.npy'), np.load('./database/QM9_deepchem/train_features.npy'))
validation_graphs = (np.load('./database/QM9_deepchem/valid_adj.npy'), np.load('./database/QM9_deepchem/valid_features.npy'))
testing_graphs = (np.load('./database/QM9_deepchem/test_adj.npy'), np.load('./database/QM9_deepchem/test_features.npy'))

# Ground truth Properties
training_prop = pd.read_csv('./database/QM9_deepchem/train_data.csv')[prop]
validation_prop = pd.read_csv('./database/QM9_deepchem/val_data.csv')[prop]
testing_prop = pd.read_csv('./database/QM9_deepchem/test_data.csv')[prop]

# For tensorboard logs (Propably dont need anymore)
log_dir = f'save2/logs/{model_name}'
# summary_writer = tf.summary.FileWriter(log_dir, model.sess.graph)

#For Visualization
val_mse_per_epoch = []
val_mae_per_epoch = []
learning_rates = []

TRAIN_ITERATIONS = int(len(training_graphs[0]) / batch_size)
VAL_ITERATIONS = int(len(validation_graphs[0]) / batch_size)
TEST_ITERATIONS = int(len(testing_graphs[0]) / batch_size)

print("Begin Training:")

# Training and validation loop
for epoch in range(num_epochs):
    # Learning rate scheduling
    new_learning_rate = (learning_rate * (decay_rate ** epoch))
    model.assign_lr(new_learning_rate)
    learning_rates.append(new_learning_rate)
    
    train_total_loss = 0


    for i in range(TRAIN_ITERATIONS):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        train_A = training_graphs[0][start_idx: end_idx]
        train_X = training_graphs[1][start_idx: end_idx]
        train_P = training_prop[start_idx: end_idx]

        loss = model.train(train_A, train_X, train_P)
        train_total_loss += loss
        # _graph, _property = loadData('./database/QM9_deepchem', 'training_dataset', i, FLAGS.unitLen, prop) #loadInputs(FLAGS, i, model_name, FLAGS.unitLen)
        # num_batches = int(_graph[0].shape[0]/batch_size)

        # Iterate over batches
        # for batch_idx in range(num_batches):
            
        #     train_batch_A = _graph[0][start_idx: end_idx]
        #     train_batch_X = _graph[1][start_idx: end_idx]
        #     train_batch_P = _property[start_idx:end_idx]
            
        #     # Train the model on the batch
        #     loss = model.train(train_batch_A, train_batch_X, train_batch_P)
        #     train_total_loss += loss
        #     print(loss)
        #     sys.exit(1)

    val_total_loss = 0
    val_preds = []
    val_targets = []
    for i in range(VAL_ITERATIONS):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        val_A = validation_graphs[0][start_idx: end_idx]
        val_X = validation_graphs[1][start_idx: end_idx]
        val_P = validation_prop[start_idx: end_idx]

        val_pred, val_loss = model.test(val_A, val_X, val_P)
        val_preds.append(val_pred)
        val_targets.append(val_P)
        val_total_loss += val_loss
        # _graph, _property = loadData('./database/QM9_deepchem', 'validation_dataset', i, FLAGS.unitLen, prop) #loadInputs(FLAGS, i, model_name, unit_len)
        # num_batches = int(_graph[0].shape[0]/batch_size)

        # Iterate over batches
        # for batch_idx in range(num_batches):
        #     start_idx = batch_idx * batch_size
        #     end_idx = (batch_idx + 1) * batch_size
        #     val_batch_A = _graph[0][start_idx: end_idx]
        #     val_batch_X = _graph[1][start_idx: end_idx]
        #     val_batch_P = _property[start_idx:end_idx]

        #     # Train the model on the batch
        #     val_pred, val_loss = model.test(val_batch_A, val_batch_X, val_batch_P)
        #     val_preds.append(val_pred)
        #     val_targets.append(val_batch_P)
        #     val_total_loss += val_loss

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_preds = val_preds.flatten()
    val_targets = val_targets.flatten()
    mse = np.mean(np.square(val_targets - val_preds))
    mae = np.mean(np.abs(val_targets - val_preds))
    val_mse_per_epoch.append(mse)
    val_mae_per_epoch.append(mae)
    val_preds = []
    val_targets = []

    # mae_summary = tf.Summary()
    # mae_summary.value.add(tag='validation_mae', simple_value=mae)
    # mse_summary = tf.Summary()
    # mse_summary.value.add(tag='validation_mse', simple_value=mse)
    # summary_writer.add_summary(mae_summary, global_step=epoch)
    # summary_writer.add_summary(mse_summary, global_step=epoch)

    # Save the model
    ckpt_path = f'save2/{model_name}/{model_name}.ckpt'
    model.save(ckpt_path, epoch)
    # Print training progress
    print("Epoch {}: Training Loss = {:.4f}, Validation Loss: MSE = {:.4f}, MAE = {:.4f}".format(epoch+1, train_total_loss/TRAIN_ITERATIONS, mse, mae))

plot_loss("Epoch", save_path=f"results/{model_name}-validate.png", title="Validation Loss", mae=val_mae_per_epoch, mse=val_mse_per_epoch)
plot_loss("Epoch", save_path=f"results/{model_name}-lr.png", title="Dynamic Learning Rate Schedule", learning_rate=learning_rates)

print("Begin Testing: ")

# Testing the model
test_total_loss = 0
test_preds = []
test_targets = []
test_loss_batch = []
for i in range(TEST_ITERATIONS):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    test_A = testing_graphs[0][start_idx: end_idx]
    test_X = testing_graphs[1][start_idx: end_idx]
    test_P = testing_prop[start_idx: end_idx]

    test_pred, test_loss = model.test(test_A, test_X, test_P)
    test_preds.append(test_pred)
    test_targets.append(test_P)
    test_total_loss += test_loss
    test_loss_batch.append(test_loss)
    # _graph, _property = loadData('./database/QM9_deepchem', 'testing_dataset', i, FLAGS.unitLen, prop) #loadInputs(FLAGS, i, model_name, unit_len)
    # test_num_batches = int(_graph[0].shape[0]/batch_size)

    # for batch_idx in range(test_num_batches):
    #     start_idx = batch_idx * batch_size
    #     end_idx = (batch_idx + 1) * batch_size
    #     test_batch_A = _graph[0][start_idx:end_idx]
    #     test_batch_X = _graph[1][start_idx:end_idx]
    #     test_batch_P = _property[start_idx:end_idx]
            
    #     # Test the model on the training batch
    #     test_pred, test_loss = model.test(test_batch_A, test_batch_X, test_batch_P)
    #     test_preds.append(test_pred)
    #     test_targets.append(test_batch_P)
    #     test_total_loss += test_loss
    #     test_loss_batch.append(test_loss)

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_preds = test_preds.flatten()
test_targets = test_targets.flatten()
mse = np.mean(np.square(test_targets - test_preds))
mae = np.mean(np.abs(test_targets - test_preds))


# print(test_targets[0])
# Print testing results
print("Testing Loss = {:.4f}, MSE = {:.4f}, MAE = {:.4f}".format(test_total_loss/TEST_ITERATIONS, mse, mae))

    
# max_value = np.max(test_loss_batch)
# max_index = np.argmax(test_loss_batch)

# print(max_index, max_value)

plot_loss("Batch", save_path=f"results/{model_name}-test.png", test_loss=test_loss_batch)
plot_loss("Molecule Number", save_path=f"results/{model_name}-labels_vs_preds.png", test_targets=test_targets[50:100], test_preds=test_preds[50:100])

# summary_writer.close()