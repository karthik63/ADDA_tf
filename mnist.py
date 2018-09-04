from __future__ import print_function
import tensorflow as tf
import numpy as np

classes=5

class Mnist:

    #the images are already normalized between 0 and 1
    def __init__(self,imageSize=28,channels=1):
        
        self.imageSize = imageSize
        self.channels = channels
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        
        train_labels = mnist.train.labels
        train_data = mnist.train.images
        
        data=[]
        labels=[]

        ## can be improved
        ##processing to get only labelled data for 5-9
        for i in range(0,train_labels.shape[0]):
            if train_labels[i]>=5 and train_labels[i]<10:
                data.append(train_data[i])
                temp=[0]*classes
                temp[train_labels[i]-classes]=1
                labels.append(temp)

        train_labels = mnist.validation.labels
        train_data = mnist.validation.images

        for i in range(0,train_labels.shape[0]):
            if train_labels[i]>=5 and train_labels[i]<10:
                data.append(train_data[i])
                temp=[0]*classes
                temp[train_labels[i]-classes]=1
                labels.append(temp)
        
        self.train_images = np.array(data)
        self.train_labels = np.array(labels)

        
        #similarly repeating for test data
        test_data = mnist.test.images
        test_labels = mnist.test.labels # Returns np.array

        data=[]
        labels=[]
        ##processing to get only labelled data for 5-9
        for i in range(0,test_labels.shape[0]):
            if test_labels[i]>=5 and test_labels[i]<10:
                data.append(test_data[i])
                temp=[0]*classes
                temp[test_labels[i]-classes]=1
                labels.append(temp)
        

        self.test_images = np.array(data)
        self.test_labels = np.array(labels)

        #why
        self.test_images = np.reshape(self.test_images,(-1,self.imageSize,self.imageSize,self.channels))


        print ("MNIST data size")
        print (self.train_images.shape,self.train_labels.shape)
        print (self.test_images.shape,self.test_labels.shape)

        #to get each batch of data every epoch
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.num_examples = self.train_images.shape[0]

    def next_batch(self, batch_size, labelled=False):
        start = self.index_in_epoch
        
        if start == 0 and self.epochs_completed == 0:
            
            idx = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexes
            self.train_images = self.train_images[idx]  # get list of `num` random samples
            self.train_labels = self.train_labels[idx]

        # go to the next epoch
        if start + batch_size > self.num_examples:
            
            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            data_rest_part = self.train_images[start:self.num_examples]
            labels_rest_part = self.train_labels[start:self.num_examples]
            idx0 = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            
            self.train_images = self.train_images[idx0]  # get list of `num` random samples
            self.train_labels = self.train_labels[idx0]

            start = 0
            self.index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self.index_in_epoch  
            data_new_part =  self.train_images[start:end]  
            labels_new_part = self.train_labels[start:end]
            data = np.reshape(np.concatenate((data_rest_part, data_new_part), axis=0),(batch_size,self.imageSize,self.imageSize,self.channels))
            labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            
            data = np.reshape(self.train_images[start:end],(batch_size,self.imageSize,self.imageSize,self.channels))
            labels = self.train_labels[start:end]

        if labelled==True:
            return data,labels
        else:
            return data

    def next_batch_test(self, batch_size, labelled=False):

        self.num_examples = self.test_images.shape[0]

        start = self.index_in_epoch

        if start == 0 and self.epochs_completed == 0:
            idx = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexes
            self.test_images = self.test_images[idx]  # get list of `num` random samples
            self.test_labels = self.test_labels[idx]

        # go to the next epoch
        if start + batch_size > self.num_examples:

            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            data_rest_part = self.test_images[start:self.num_examples]
            labels_rest_part = self.test_labels[start:self.num_examples]
            idx0 = np.arange(0, self.num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes

            self.test_images = self.test_images[idx0]  # get list of `num` random samples
            self.test_labels = self.test_labels[idx0]

            start = 0
            self.index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self.index_in_epoch
            data_new_part = self.test_images[start:end]
            labels_new_part = self.test_labels[start:end]
            data = np.reshape(np.concatenate((data_rest_part, data_new_part), axis=0),
                              (batch_size, self.imageSize, self.imageSize, self.channels))
            labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:

            self.index_in_epoch += batch_size
            end = self.index_in_epoch

            data = np.reshape(self.test_images[start:end], (batch_size, self.imageSize, self.imageSize, self.channels))
            labels = self.test_labels[start:end]

        if labelled == True:
            return data, labels
        else:
            return data
