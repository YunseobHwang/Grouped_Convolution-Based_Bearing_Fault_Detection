#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime
import itertools
from sklearn.metrics import confusion_matrix


# # Load Data

# In[2]:


def load_data_and_label(data_dir, target_str = '', reshape_need = False, extend = False):
    
    """
    data_dir: A folder including npy.files.
    target_str: When there are various type of data, you can sort data by the desired type with 'target_str'
    """
    
    npy_files = sorted(os.listdir(data_dir))
    if target_str != '':
        npy_files = [i for i in npy_files if target_str in i]

    normal = np.load(os.path.join(data_dir, str([i for i in npy_files if 'normal' in i][0])))
    ball_7 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'ball_7' in i][0])))
    ball_14 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'ball_14' in i][0])))
    ball_21 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'ball_21' in i][0])))
    inner_7 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'inner_7' in i][0])))
    inner_14 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'inner_14' in i][0])))
    inner_21 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'inner_21' in i][0])))
    outer_7 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'outer_7' in i][0])))
    outer_14 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'outer_14' in i][0])))
    outer_21 = np.load(os.path.join(data_dir, str([i for i in npy_files if 'outer_21' in i][0])))

    normal_y = one_hot(normal, 0, nb_classes = 10)
    ball_7_y = one_hot(ball_7, 1, nb_classes = 10)
    ball_14_y = one_hot(ball_14, 2, nb_classes = 10)
    ball_21_y = one_hot(ball_21, 3, nb_classes = 10)
    inner_7_y = one_hot(inner_7, 4, nb_classes = 10)
    inner_14_y = one_hot(inner_14, 5, nb_classes = 10)
    inner_21_y = one_hot(inner_21, 6, nb_classes = 10)
    outer_7_y = one_hot(outer_7, 7, nb_classes = 10)
    outer_14_y = one_hot(outer_14, 8, nb_classes = 10)
    outer_21_y = one_hot(outer_21, 9, nb_classes = 10)

    test_x = np.vstack([normal, ball_7, ball_14, ball_21, inner_7, inner_14, inner_21, outer_7, outer_14, outer_21])
    labels = np.vstack([normal_y, ball_7_y, ball_14_y, ball_21_y, inner_7_y, inner_14_y, inner_21_y, outer_7_y, outer_14_y, outer_21_y])

    if reshape_need != False:
        test_x = reshapeAsimage(test_x)
    if extend != False:
        test_x = np.expand_dims(test_x, axis = 3)
    print(test_x.shape, labels.shape)
    return test_x, labels


# In[3]:


def one_hot(data, classes, nb_classes = 2):
    one_hot = [0]*nb_classes
    one_hot[classes] = 1
    return np.vstack([one_hot for i in range(len(data))])


def reshapeAsimage(data):
    """
    inputshape = (Number, Channel, Height(f), Width(t))
    outputshape = (Number, Height(f), Width(t), Channel)
    """
    N, H, W, C = data.shape[0], data.shape[2], data.shape[3], data.shape[1]
    reshape = np.zeros([N, H, W, C])
    for n in range(N):
        for f in range(H):
            for t in range(W):
                for c in range(C):
                    reshape[n, f, t, c] = data[n, c, f, t]
    return reshape


# # Training

# In[4]:



def train_valid_split(data, label, train_rate = 0.85):
    train_idx = np.sort(np.random.choice(len(data), round(len(data)*train_rate), replace = False))
    valid_idx = np.setxor1d(train_idx, np.arange(len(data)))
    return data[train_idx], label[train_idx], data[valid_idx], label[valid_idx]

def random_minibatch(x, y, batch_size = 50):
    idx = np.random.choice(len(x), batch_size)
    return x[idx], y[idx]

def shuffle(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    if type(x) == type(y):
        return x[idx], y[idx] 
    else:
        return x[idx]
    



# In[5]:


class training_history:
    def __init__(self, accr_train, accr_valid, loss_train, loss_valid):
        self.accr_train = accr_train
        self.accr_valid = accr_valid
        self.loss_train = loss_train
        self.loss_valid = loss_valid
    def table(self):
        print('==============================================================')
        print('[Iter] || Train_accr || Valid_accr || Train_loss || Valid_loss')
        print('==============================================================')
    def evl(self, n_iter):
        evl = '[{:*>4d}] || {:*>.2f} %    || {:*>.2f} %    || {:.8f} || {:.8f}'.format(n_iter, 
                                                                                      self.accr_train[-1]*100, self.accr_valid[-1]*100, 
                                                                                      self.loss_train[-1], self.loss_valid[-1])
        return evl
    def prt_evl(self, n_iter):
        print(self.evl(n_iter))
        print('--------------------------------------------------------------')
    def early_under(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping - Underffiting !!]\n')
    def early_over(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping - Overffiting !!]\n')
    def early(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping]\n')
    def done(self, n_iter, train_time, early_stopping):  
        global training_name
        global contents
        global filename
        global title
        
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%y%m%d%H%M')
        
        contents = (
        'Training Time : {} Min.\n'.format(train_time) +
        'Early Stopping : {}\n'.format(early_stopping) +
        'Iteration : {}\n'.format(n_iter)
        )
        print(contents)

        title = 'Training History'
    def plot(self, n_cal):
        fig = plt.figure(figsize = (15,20))
        plt.suptitle('Training History', y = 0.92, fontsize = 20)

        x_axis = range(1, len(self.accr_train)+1)

        plt.subplot(2, 1, 1)
        plt.plot(x_axis, self.accr_train, 'b-', label = 'Training Accuracy')
        plt.plot(x_axis, self.accr_valid, 'r-', label = 'Validation Accuracy')
        plt.xlabel('n_iter/n_cal (n_cal = {})'.format(n_cal), fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.legend(fontsize = 10)
        plt.subplot(2, 1, 2)
        plt.plot(x_axis, self.loss_train, 'b-', label = 'Training Loss')
        plt.plot(x_axis, self.loss_valid, 'r-', label = 'Validation Loss')
        plt.xlabel('n_iter/n_cal (n_cal = {})'.format(n_cal), fontsize = 15)
        plt.ylabel('Loss', fontsize = 15)
    #     plt.yticks(np.arange(0, 0.25, step=0.025))
        plt.legend(fontsize = 12)
        plt.show()


# # Testing

# In[6]:


class ResNet:  
    def __init__(self, model_path):
        self.test_graph = tf.Graph()
        with self.test_graph.as_default():
            self.sess = tf.Session(graph = self.test_graph)
            loader = tf.train.import_meta_graph(model_path + '.meta')
            loader.restore(self.sess, model_path)

            self.bn_prob = self.test_graph.get_tensor_by_name('bn_prob:0')
            self.x = self.test_graph.get_tensor_by_name('img:0')
            self.score = self.test_graph.get_tensor_by_name('dense/BiasAdd:0')

    def get_softmax(self, image):
        softmax = self.sess.run(tf.nn.softmax(self.score), feed_dict={self.x: image, self.bn_prob: False})
        return softmax
    
class VGGNet:  
    def __init__(self, model_path):
        self.test_graph = tf.Graph()
        with self.test_graph.as_default():
            self.sess = tf.Session(graph = self.test_graph)
            loader = tf.train.import_meta_graph(model_path + '.meta')
            loader.restore(self.sess, model_path)

            self.is_training = self.test_graph.get_tensor_by_name('is_training:0')
            self.x = self.test_graph.get_tensor_by_name('img:0')
            self.score = self.test_graph.get_tensor_by_name('dense_2/BiasAdd:0')

    def get_softmax(self, image):
        softmax = self.sess.run(tf.nn.softmax(self.score), feed_dict={self.x: image, self.is_training: False})
        return softmax


# In[7]:


def test_batch_idxs(data, batch_size = None):
    """generate the serial batch of data on index-level.
       Usually, the data is too large to be evaluated at once.
    
    Args:
      data: A list or array of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch_idxs: A list, 
    """
    if batch_size == None:
        batch_size = 250
    
    total_size = len(data)
    batch_idxs = []
    start = 0
    while True:
        if total_size >= start + batch_size:
            batch_idxs.append([start + i for i in range(batch_size)])
        elif total_size < start + batch_size:
            batch_idxs.append([start + i for i in range(total_size - start)])
        start += batch_size
        if total_size <= start:
            break
    return batch_idxs

def batch_flatten(data):
    """flatten A list stacked with the result of batch.
    
    Args:
      data: A list or array  
      
    Returns:
      Data: A list, total result
    """
    batch_n = len(data)
    for i in range(batch_n):
        if i == 0:
            Data = data[i]
        else:
            Data = np.concatenate((Data, data[i]), axis = 0)   
    return Data

def model_pred(Model, image, n_batch = None):
    b_idxs = test_batch_idxs(image, batch_size = n_batch)
    outputs = []
    start_time = time.time()
    for b_idx in b_idxs:
        softmax = Model.get_softmax(image[b_idx])
        outputs.append(softmax)
    time_taken = time.time() - start_time
    print("Inference Time:",time.strftime("%H:%M:%S", time.gmtime(time_taken)))
    outputs = batch_flatten(outputs)
    return outputs


# In[8]:


def accuracy(data_y, outputs):
    pred = np.argmax(outputs, axis = 1)
    true = np.argmax(data_y, axis = 1)
    acc = 100*np.mean(np.equal(true, pred))
    print("Accuracy: {:.2f} %".format(acc))
    return acc

def compute_accr_std(data_y, outputs):
    pred = np.argmax(outputs, axis = 1)
    true = np.argmax(data_y, axis = 1)
    accr = np.equal(true, pred)
    avg = 100*np.mean(accr)
    std = 100*np.std(accr)
    
    print("Accuracy: {:.2f} +/- {:.2f}%".format(avg, std))
    return avg, std

def compute_accr_sem(data_y, outputs):
    pred = np.argmax(outputs, axis = 1)
    true = np.argmax(data_y, axis = 1)
    accr = np.equal(true, pred)
    avg = 100*np.mean(accr)
    sem= 100*stats.sem(accr)
    
    print("Accuracy: {:.2f} +/- {:.2f}%".format(avg, sem))
    return avg, sem

def con_mat(data_y, outputs):
    pred = np.argmax(outputs, axis = 1)
    true = np.argmax(data_y, axis = 1)
    return confusion_matrix(true, pred)

def plot_con_mat(cm, value_size = 15, label_size = 10, mode = 'percent'):
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if mode == 'percent':
            value = np.round(cm[i, j]/(np.sum(cm, 1)[i]), 3)
        if mode == 'num':
            value = cm[i, j]
        plt.text(j, i, value,
                 fontsize = value_size,
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label', fontsize = label_size)
    plt.xlabel('Predicted', fontsize = label_size)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
               ['nor', 'ball_7', 'ball_14', 'ball_21', 'inner_7', 'inner_14', 'inner_21', 'outer_7', 'outer_14', 'outer_21'], 
               rotation=-90, fontsize = label_size)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
               ['nor', 'ball_7', 'ball_14', 'ball_21', 'inner_7', 'inner_14', 'inner_21', 'outer_7', 'outer_14', 'outer_21'], 
               rotation=0, fontsize = label_size)

