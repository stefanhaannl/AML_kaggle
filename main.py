"""
Main working directory
"""
import preprocessing
import tensorflow as tf
import numpy as np

class DataFile():
    
    def __init__(self,n=0):
        self.traindata, self.trainlabels, self.testdata, self.testlabels = self.load_data(n)
        self.inputdimension = list(self.traindata.shape)
        self.outputdimension = list(self.trainlabels.shape)
        self.n_labels = self.outputdimension[1]
        
    def load_data(self,n=0,train=True):
        df = preprocessing.load_images(n,train)
        df = preprocessing.resize_images(df)
        if train == True:
            df = preprocessing.add_labels(df)
            df = preprocessing.add_label_hotmap(df)
        x_train, x_test, y_train, y_test = preprocessing.split_data(df)
        x_train = np.swapaxes(np.dstack(np.array(x_train)),0,2)
        x_test = np.swapaxes(np.dstack(np.array(x_test)),0,2)
        return x_train, np.array(list(y_train)), x_test, np.array(list(y_test))
    
    def get_batch(self,batchsize):
        ind = np.random.randint(self.traindata.shape[0],size=batchsize)
        batch_x = self.traindata[ind,:]
        batch_y = self.trainlabels[ind,:]
        return batch_x, batch_y
    
    def get_testbatch(self,batchsize):
        ind = np.random.randint(self.testdata.shape[0],size=batchsize)
        batch_x = self.testdata[ind,:]
        batch_y = self.testlabels[ind,:]
        return batch_x, batch_y


class Network():
    """
    Netwerk structure class. To work with the network:
        1. Create a new network object and pass in the DataFile object.
        2. Add layers in order by calling the layer_add_... methods.
        3. OPTIONAL: Change the parameters by calling the set_parameters method.
        4. Train the model by calling the train method.
    """
    def __init__(self,datafile):
        """
        Creates a Neural network with tensorflow, specify a DataFile object.
        """
        #LOAD THE DATA
        self.datafile = datafile
        self.datalength = self.datafile.inputdimension[0]
        #SET PARAMETERS
        self.set_parameters()
        #DEFINE PLACEHOLDERS
        self.x = tf.placeholder(tf.float32,[None]+self.datafile.inputdimension[1:])
        self.y_true = tf.placeholder(tf.float32,[None]+self.datafile.outputdimension[1:])
        #BUILD THE BODY OF THE NEURAL NETWORK
        self.initialize_structure()
        #DEFINE THE LOSS FUNCTION
        self.define_loss_function()
        
    def set_parameters(self,session_steps=5000,batch_size=50,learning_rate=0.001,dropout_hold_prob=0.5):
        """
        Variables:
            session_steps: (default:5000)
            batch_size: (default:50)
            learning_rate: (default: 0.001)
            dropout_hold_prob: (default: 0.5)
        """
        self.session_steps = session_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_hold_prob = dropout_hold_prob
        
    def initialize_structure(self):
        """
        Creates a list of layers where all layers of the model are saved. Also adds a default output layer.
        """
        self.hold_prob = tf.placeholder(tf.float32)
        self.layers = [OutputLayer(self.datafile.inputdimension[1:],self.datafile.n_labels,self.hold_prob)]
        self.calculate_operations()
        
    def calculate_operations(self):
        """
        Every time the network structure is changed, this method is called in order to redifine self.y. Automatically prints the changed model.
        """
        self.y = self.x
        for layer in self.layers:
            self.y = layer.forward(self.y)
        self.define_loss_function()
        print('The model has been changed, an overview from input to output:\n\n')
        self.print_model()
            
    def print_model(self):
        """
        Prints the structure of the current model to the console.
        """
        for layer in self.layers:
            if isinstance(layer,NormLayer):
                print('Layer: Normal Fully Connected (Nodes:',layer.nodes,', Transfer Function:',layer.transfer_func,')')
            elif isinstance(layer,ConvLayer):
                print('Layer: Convolutional (Kernel Chanels:',layer.channels,', Kernel Size:',layer.size,', Transfer Function:',layer.tranfer_func,')')
            elif isinstance(layer,PoolLayer):
                print('Layer: Pooling /2')
            else:
                print('Layer: Output Fully Connected')
            print('Input shape: ',layer.inputshape)
            print('Output shape: ',layer.outputshape)
            print('\n')
        
    def layer_add_normal(self,n=1024,transfer_function = tf.nn.relu):
        """
        Adds a normal fully connected layer, specify the layer parameters:
            n: The number of neurons in the layer (default: 1024)
            transfer_func: The tranfer function used for the layer (default: tf.nn.relu)
        """
        if len(self.layers) == 1:
            input_shape = self.datafile.inputdimension[1:]
        else:
            input_shape = self.layers[-2].outputshape
        self.add_layer(NormLayer(input_shape,n,transfer_function))
        
    def layer_add_convolutional(self, channels=32, W=6, H=6, transfer_function = tf.nn.relu):
        """
        Adds a convolutional layer, specify the layer parameters:
            channels: The number of output channels (default: 32)
            W: The width of the kernel (default: 6)
            H: The height of the kernel (default: 6)
            transfer_func: The transfer function used for the layer (default: tf.nn.relu)
        """
        if len(self.layers) == 1:
            input_shape = self.datafile.inputdimension[1:]
        else:
            input_shape = self.layers[-2].outputshape
        self.add_layer(ConvLayer(input_shape,channels,[W,H],transfer_function))
        
    def layer_add_pooling(self):
        """
        Adds a pooling layer, specify the layer parameters:
            SIZE CANNOT BE SPECIFIED YET
            CURRENT SIZE IS 2 (A SPLIT OF N/2)
        """
        if len(self.layers) == 1:
            input_shape = self.datafile.inputdimension[1:]
        else:
            input_shape = self.layers[-2].outputshape
        self.add_layer(PoolLayer(input_shape))
        
    def add_layer(self,layer):
        del self.layers[-1]
        self.layers.append(layer)
        self.layers.append(OutputLayer(self.layers[-1].outputshape,self.datafile.n_labels,self.hold_prob))
        self.calculate_operations()
        
    def define_loss_function(self):
        """
        Defines the loss function an creates a trainer variable for in the session.
        """
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.trainer = optimizer.minimize(cross_entropy)
    
    def train(self):
        """
        Train the current model and evaluate it on the testing data.
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for step in range(self.session_steps):
                batch_x, batch_y = self.datafile.get_batch(self.batch_size)
                sess.run(self.trainer,feed_dict = {self.x:batch_x, self.y_true:batch_y, self.hold_prob:self.dropout_hold_prob})
                if step%100 == 0:
                    print("Running step", step, "/",self.session_steps)
                    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_true,1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                    feed_x, feed_y = self.datafile.get_testbatch(1000)
                    print('Accuracy:',sess.run(accuracy,feed_dict={self.x:feed_x,self.y_true:feed_y,self.hold_prob:1.0}))
                    print('\n')

"""
###############################################################################
MODEL COMPONENTS
###############################################################################
"""
        
class Layer():
    
    def __init__(self, inputshape, outputshape):
        self.inputshape = inputshape
        self.outputshape = outputshape
    
    def init_weights(self,shape):
        init_random_dist = tf.truncated_normal(shape,stddev=1/np.prod(shape))
        return tf.Variable(init_random_dist)
    
    def init_bias(self,shape):
        init_bias_vals = tf.truncated_normal(shape,stddev=1/np.prod(shape))
        return tf.Variable(init_bias_vals)
    
    
class OutputLayer(Layer):
    
    def __init__(self,inputshape,outputshape,dropout):
        self.dropout = dropout
        super().__init__(inputshape,[outputshape])
        self.inputlength = np.prod(inputshape)
        self.W = self.init_weights([self.inputlength,outputshape])
        self.b = self.init_bias([outputshape])
    
    def forward(self,input_layer):
        shape = input_layer.get_shape().as_list()
        if len(shape) > 2:
            dim = np.prod(shape[1:])
            input_layer = tf.reshape(input_layer,[-1,dim])
        layer = tf.nn.dropout(input_layer,self.dropout)
        return tf.add(tf.matmul(layer,self.W),self.b)


class NormLayer(Layer):
    def __init__(self,inputshape,n=1024,transfer_func=tf.nn.relu):
        super().__init__(inputshape,[n])
        self.nodes = n
        self.transfer_func = transfer_func
        self.inputlength = np.prod(inputshape)
        self.W = self.init_weights([self.inputlength,n])
        self.b = self.init_bias([n])
        
    def forward(self,input_layer):
        shape = input_layer.get_shape().as_list()
        if len(shape) > 2:
            dim = np.prod(shape[1:])
            input_layer = tf.reshape(input_layer,[-1,dim])
        return self.transfer_func(tf.add(tf.matmul(input_layer,self.W),self.b))


class ConvLayer(Layer):
    
    def __init__(self,inputshape,channels,size,tranfer_func=tf.nn.relu):
        self.channels = channels
        self.size = size
        self.tranfer_func = tranfer_func
        if len(inputshape) == 2:
            self.kernel = self.size+[1]+[self.channels]
        else:
            self.kernel = self.size+[inputshape[2]]+[inputshape[2]+self.channels]
        outputshape = [inputshape[0],inputshape[1],self.kernel[3]]
        super().__init__(inputshape,outputshape)
        self.W = self.init_weights(self.kernel)
        self.b = self.init_bias([self.kernel[3]])
        
    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
    def forward(self,input_layer):
        if len(input_layer.shape) == 3:
            input_layer = tf.reshape(input_layer,[-1,int(input_layer.shape[1]),int(input_layer.shape[2]),1])
        return self.tranfer_func(self.conv2d(input_layer,self.W))
    

class PoolLayer(Layer):
    
    def __init__(self,inputshape):
        outputshape = inputshape[:]
        outputshape[0] = int(np.ceil(outputshape[0]/2))
        outputshape[1] = int(np.ceil(outputshape[1]/2))
        super().__init__(inputshape,outputshape)
    
    def max_pool_2by2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    def forward(self,input_layer):
        if len(input_layer.shape) == 3:
            input_layer = tf.reshape(input_layer,[-1,int(input_layer.shape[1]),int(input_layer.shape[2]),1])
        return self.max_pool_2by2(input_layer)
