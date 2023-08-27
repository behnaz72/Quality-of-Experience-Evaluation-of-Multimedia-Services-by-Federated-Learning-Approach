#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
from typing import List, Any
import cv2
import random
import metr as metr
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[2]:


class CNN:
    def build(self):
        num_classes = 5
        model = Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', name='layer_1', input_shape=(112, 112, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', name='layer_2'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='layer_3'))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', name='layer_4'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', name='layer_5'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='layer_6'))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(256, (3, 3), activation='relu', name='layer_7'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), activation='relu', name='layer_8'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='layer_9'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten(name='layer_10'))
        model.add(layers.Dense(512, activation='relu', name='layer_11'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax', name='layer_12'))
        model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learning_rate,
                                                                      decay=learning_rate / comms_round,
                                                                      momentum=0.9),
                      metrics=['accuracy'])
        return model



# In[3]:


# Model compile parameters
batch = 256
client_number = 6
train_test_data_size = 2400
global_count = 0
local_count = []
comms_round = 10
learning_rate = 0.01


# In[4]:


class Client:
    def __init__(self):
        self.ID = random.randint(0, 10000)
        self.dataNum = {}
        print(self.ID)

    def returnID(self):
        return self.ID

    def train_client(self, number, global_model):
        global global_count, local_count
        if number == 0:
            number = self.dataNum[self.ID]
            print("Number from dictionary is: %d" % number)
        else:
            self.dataNum[self.ID] = number
            print("Number from function is: %d" % number)
        Data_path = '/users/behnaz/Desktop/dataset/client batches/Client'
        Data = self.load_Data(Data_path, number)
        X_train, X_test = self.change_shape(Data, train_test_data_size)

        Label_path = "/users/behnaz/Desktop/dataset/labels/MOS_CL"
        labels = self.load_Label(Label_path, number)

        Y_train = labels[1:train_test_data_size + 1]
        Y_train = Y_train[:, 1]
        one_hot_Y_train = self.label_normalize(Y_train)

        Y_test = labels[train_test_data_size + 1:]
        Y_test = Y_test[:, 1]
        one_hot_Y_test = self.label_normalize(Y_test)

        CNN_Local = CNN()
        local_Model = CNN_Local.build()
        local_Model.set_weights(global_model.get_weights())
        history = local_Model.fit(X_train, one_hot_Y_train, epochs=7000, batch_size=batch)
        results = local_Model.evaluate(X_test, one_hot_Y_test)
        accuracy = results[1]
        local_count.append(len(X_train))
        global_count += len(X_train)
        name = "model" + str(self.ID) + ".hdf5"
        local_Model.save_weights('/users/behnaz/Desktop/dataset/models/' + name)
        return accuracy
    def load_Data(self, path, num):
        Data = []
        dataPath = path + str(num)
        for myData in os.listdir(dataPath):
            img = cv2.imread(os.path.join(dataPath, myData))
            Data.append(img)
        Data = np.array(Data)
        return Data
    def load_Label(self, path, num):
        labels = open(path + str(num) + '.csv', "r")
        csv_reader = csv.reader(labels)
        lists_from_csv = []
        for row in csv_reader:
            lists_from_csv.append(row)
        labels = np.array(lists_from_csv)
        return labels
    def change_shape(self, data, index):
        train = data[:index]
        #train = train.astype('float32') / 255.0
        #test = data[index:] / 255.0
        x = train
        train = (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)
        test = data[index:]
        x = test
        test = (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)
        return train, test
    def label_normalize(self, data):
        rounded_Data = [round(float(num)) for num in data]
        mymin = min(rounded_Data)
        mymax = max(rounded_Data)
        scaled_Data = [(4 * (x - mymin) / (mymax - mymin)) + 1 for x in rounded_Data]
        round_Data = [round(float(num)) for num in scaled_Data]
        round_Data = [x - 1 for x in round_Data]
        one_hot_Data = to_categorical(round_Data)
        return one_hot_Data
class Server:
    def __init__(self, clientNum, train_test_data_size):
        self.clientNum = clientNum
        self.train_test_data_size = train_test_data_size
        self.clientDict = {}
        self.model_Model = None
        self.global_Model = None
        self.runCounter = 0
    def create_clients(self, number):
        self.clientDict = {}
        for x in range(number):
            self.clientDict[x + 1] = Client()
    def data_partitioning(self, number):
        partition = {}
        size = int(number / self.clientNum)
        for i in range(self.clientNum - 1):
            partition[i + 1] = size
            print(partition[i + 1])
        partition[self.clientNum] = number - (size * (self.clientNum - 1))
        print(partition[self.clientNum])

        for i in range(self.clientNum):
            self.clientDict[i + 1].dataNum = partition
    def draw_accuracy_chart(self, client_accuracy):
        plt.figure()
        plt.bar(range(self.clientNum), client_accuracy)
        plt.xlabel('Client')
        plt.ylabel('Accuracy')
        plt.title('Accuracy across Clients')
        plt.show()
    def train_clients(self, comms_round):
        client_accuracy = []
        global global_count
        global_count = 0
        global local_count
        local_count = []
        self.global_Model = CNN().build()
        for i in range(self.clientNum):
            accuracy = self.clientDict[i + 1].train_client(i + 1, self.global_Model)
            client_accuracy.append(accuracy)
        self.draw_accuracy_chart(client_accuracy)
    def weight_scalling_factor(self, cnt):
        global_cnt = global_count
        local_cnt = cnt
        return local_cnt / global_cnt
    def scale_model_weights(self, weight, scalar):
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final
    def sum_scaled_weights(self, scaled_weight_list):
        avg_grad = list()
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad
    def test_model(self, X_test, Y_test, model, comm_round):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        logits = model.predict(X_test)
        loss = cce(Y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
        print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
        return acc, loss, logits
    def create_global_model(self):
        counter = 0
        for comm_round in range(comms_round):
            global_weights = self.global_Model.get_weights()
            Data_path = '/users/behnaz/Desktop/dataset/models/'
            print(local_count)
            for myWeights in os.listdir('/users/behnaz/Desktop/dataset/models/'):
                self.model_Model = CNN().build()
                self.model_Model.load_weights(os.path.join(Data_path, myWeights))
                loc_cnt = local_count[counter % client_number]
                scaled_local_weight_list: list[list[Any]] = list()
                scaling_factor = self.weight_scalling_factor(loc_cnt)
                scaled_weights = self.scale_model_weights(self.model_Model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                counter += 1
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.global_Model.set_weights(average_weights)
    def run_test_global(self, opt=None, ls=None):
        test_Client = Client()
        X_test = test_Client.load_Data('/users/behnaz/Desktop/dataset/Global Test', "")
        labels = test_Client.load_Label("/users/behnaz/Desktop/dataset/labels/MOS_Global", "")
        Y_test = labels[1:]
        Y_test = Y_test[:, 1]
        one_hot_Y_test = test_Client.label_normalize(Y_test)
        self.global_Model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learning_rate,
                                                                      decay=learning_rate / comms_round,
                                                                      momentum=0.9),
                      metrics=['accuracy'])
        results = self.global_Model.evaluate(X_test, one_hot_Y_test)
        print(results)
        test_batched = tf.data.Dataset.from_tensor_slices((X_test, one_hot_Y_test)).batch(len(one_hot_Y_test))
        for (X_test, one_hot_Y_test) in test_batched:
            global_acc, global_loss, predicts = self.test_model(X_test, one_hot_Y_test, self.global_Model, 10)
    #def draw_accuracy_chart(self, client_accuracy):
    #    plt.figure()
    #    plt.bar(range(self.global_acc), client_accuracy)
    #    plt.xlabel('Client')
    #    plt.ylabel('Accuracy')
    #    plt.title('Accuracy Global')
    #    plt.show()        
    def run(self, trainNumber):
        if (self.runCounter == 0):
            for i in range(self.clientNum):
                cli = Client()
                client_ID = cli.returnID()
                print(client_ID)
                self.clientDict[client_ID] = cli
                cli.train_client(i + 1, self.global_Model)
        else:
            keys = random.sample(list(self.clientDict), trainNumber)
            values = [self.clientDict[k] for k in keys]
            print(values)
            for i in range(len(values)):
                values[i].train_client(0, self.global_Model)
        print(self.clientDict)
        self.runCounter += 1
        self.create_global_model()
        self.run_test_global()
        return


# In[ ]:


server = Server(client_number, train_test_data_size)
server.create_clients(client_number)
server.train_clients(comms_round)
server.run_test_global()


# In[ ]:




