"""
Main working directory
"""
import preprocessing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataFile():
    
    def __init__(self,n=0,train=True):
        if train == True:
            self.traindata, self.trainlabels, self.testdata, self.testlabels = self.load_data(n)
            self.inputdimension = list(self.traindata.shape)
            self.outputdimension = list(self.trainlabels.shape)
            self.n_labels = self.outputdimension[1]
        else:
            self.testdata, self.template, self.names = self.load_data(n,train)
        
        
    def load_data(self,n=0,train=True):
        df, template = preprocessing.load_images(n,train)
        df = preprocessing.resize_images(df)
        if train == True:
            df = preprocessing.add_labels(df)
            df = preprocessing.add_label_hotmap(df)
            x_train, x_test, y_train, y_test = preprocessing.split_data(df)
            x_train = np.swapaxes(np.dstack(np.array(x_train)),0,2)
            x_test = np.swapaxes(np.dstack(np.array(x_test)),0,2)
            return x_train, np.array(list(y_train)), x_test, np.array(list(y_test))
        else:
            x_train = df['image_data']
            names = np.array(list(df['image_number']))
            x_train = np.swapaxes(np.dstack(np.array(x_train)),0,2)
            return x_train, template, names
    
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
        self.datafile = datafile
        self.set_parameters()
        self.initialize_structure()
        
    def set_parameters(self,session_steps=2000,batch_size=50,learning_rate=0.002):
        """
        Variables:
            session_steps: (default:5000)
            batch_size: (default:50)
            learning_rate: (default: 0.001)
        """
        self.session_steps = session_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mode = ['TRAIN']
        
    def initialize_structure(self):
        """
        Creates a list of layers where all layers of the model are saved. Also adds a default output layer.
        """
        self.x = tf.placeholder(tf.float32,[None]+self.datafile.inputdimension[1:])
        self.y_true = tf.placeholder(tf.float32,[None]+self.datafile.outputdimension[1:])
        self.hold_prob = tf.placeholder(tf.float32)
        self.layers = [InputLayer(self.datafile.inputdimension[1:])]
        self.calculate_operations()
        
    def calculate_operations(self):
        """
        Every time the network structure is changed, this method is called in order to redifine self.y. Automatically prints the changed model.
        """
        self.y = self.x
        for layer in self.layers:
            self.y = layer.forward(self.y)
        print('The model has been changed, an overview from input to output:\n\n')
        self.print_model()
        if layer.outputshape == self.datafile.outputdimension[1:]:
            self.define_loss_function()
            print('This model can now be trained!')
        else:
            print('This model cannot be trained yet!')
            
    def print_model(self):
        """
        Prints the structure of the current model to the console.
        """
        for layer in self.layers:
            if isinstance(layer, InputLayer):
                print('Layer: Input Images')
            elif isinstance(layer,ConvLayer):
                print('Layer: Convolutional (Kernel Chanels:',layer.channels,', Kernel Size:',layer.size,', Transfer Function:',layer.transfer_func,')')
            elif isinstance(layer,PoolLayer):
                print('Layer: Pooling (Size:',layer.size,')')
            elif isinstance(layer,FlattenLayer):
                print('Layer: Flatten')
            elif isinstance(layer,DenseLayer):
                print('Layer: Dense (Nodes:',layer.nodes,', Transfer Function:',layer.transfer_func,')')
            elif isinstance(layer,DropoutLayer):
                print('Layer: Dropout (Rate:',layer.rate,')')
            elif isinstance(layer,Layer):
                print('Layer: Output Logits')
            print('Input Shape: ',layer.inputshape)
            print('Output Shape: ',layer.outputshape)
            print('\n')
        
    def layer_add_convolutional(self, channels=32, W=6, H=6, transfer_function = tf.nn.relu):
        """
        Adds a convolutional layer, specify the layer parameters:
            channels: The number of output channels (default: 32)
            W: The width of the kernel (default: 6)
            H: The height of the kernel (default: 6)
            transfer_function: The transfer function used for the layer (default: tf.nn.relu)
        """
        self.layers.append(ConvLayer(self.layers[-1].outputshape,channels,[W,H],transfer_function))
        self.calculate_operations()
        
    def layer_add_pooling(self,size=2):
        """
        Adds a pooling layer, specify the layer parameters:
            Size: The size of the pooling (default: 2)
        """
        self.layers.append(PoolLayer(self.layers[-1].outputshape,size))
        self.calculate_operations()
        
    def layer_add_flatten(self):
        self.layers.append(FlattenLayer(self.layers[-1].outputshape))
        self.calculate_operations()
        
    def layer_add_dense(self,nodes=1024,transfer_function = tf.nn.relu):
        """
        Adds a normal fully connected layer, specify the layer parameters:
            nodes: The number of neurons in the layer (default: 1024)
            transfer_func: The tranfer function used for the layer (default: tf.nn.tanh)
        """
        self.layers.append(DenseLayer(self.layers[-1].outputshape,nodes,transfer_function))
        self.calculate_operations()
        
    def layer_add_dropout(self,rate=0.4):
        self.layers.append(DropoutLayer(self.layers[-1].outputshape,rate,self.mode))
        self.calculate_operations()
        
    def layer_add_output(self):
        self.layers.append(Layer(self.layers[-1].outputshape,[self.datafile.outputdimension[1]]))
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
        Train the current model and evaluate it on the testing data split.
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            x = np.linspace(0,self.session_steps/100,self.session_steps/100)
            y = []
            for step in range(self.session_steps):
                batch_x, batch_y = self.datafile.get_batch(self.batch_size)
                sess.run(self.trainer,feed_dict = {self.x:batch_x, self.y_true:batch_y})
                if step%100 == 0:
                    print("Running step", step, "/",self.session_steps)
                    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_true,1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                    feed_x, feed_y = self.datafile.get_testbatch(1000)
                    self.mode[0] = 'TEST'
                    accuracy = sess.run(accuracy,feed_dict={self.x:feed_x,self.y_true:feed_y})
                    self.mode[0] = 'TRAIN'
                    y.append(accuracy)
                    print('Accuracy:',accuracy)
                    print('\n')
        y = np.array(y)
        plt.plot(x,y)
        plt.ylabel("Accuracy")
        plt.xlabel("Iteration")
        return y
    
    def evaluate(self,datafile):
        print('Mode variable set to: EVAL')
        self.mode[0] = 'EVAL' 
        with tf.Session() as sess:
            labels = sess.run(tf.argmax(self.y,axis=1), feed_dict={self.x:datafile.testdata})
            self.evaluated_data = labels
        print("The model has been evaluated on the test data an has been exported to CSV format.")
        print('Mode set back to: TRAIN')
        self.mode[0] = 'TRAIN'
        
"""
###############################################################################
MODEL COMPONENTS
###############################################################################
"""
        
class Layer():
    
    def __init__(self, inputshape, outputshape):
        self.inputshape = inputshape
        self.outputshape = outputshape
    
    def forward(self,input_layer):
        return tf.layers.dense(
                inputs = input_layer,
                units = self.outputshape[0])

class InputLayer(Layer):
    
    def __init__(self,inputshape):
        outputshape = inputshape + [1]
        super().__init__(inputshape,outputshape)
    
    def forward(self,input_layer):
        return tf.reshape(input_layer,[-1]+self.outputshape)
    
    
class ConvLayer(Layer):
    
    def __init__(self,inputshape,channels,size,transfer_func):
        outputshape = inputshape[0:2]+[channels]
        super().__init__(inputshape,outputshape)
        self.channels = channels
        self.size = size
        self.transfer_func = transfer_func
        
    def forward(self,input_layer):
        return tf.layers.conv2d(
                inputs = input_layer,
                filters = self.channels,
                kernel_size = self.size,
                padding = 'same',
                activation = self.transfer_func)
    
    
class PoolLayer(Layer):
    
    def __init__(self,inputshape,size):
        outputshape = inputshape[:]
        outputshape[0] = int(round(outputshape[0]/size))
        outputshape[1] = int(round(outputshape[1]/size))
        super().__init__(inputshape,outputshape)
        self.size = size
        
    def forward(self,input_layer):
        return tf.layers.max_pooling2d(
                inputs = input_layer,
                pool_size = [self.size,self.size],
                strides = self.size)
        

class FlattenLayer(Layer):
    
    def __init__(self,inputshape):
        outputshape = [np.prod(inputshape)]
        super().__init__(inputshape,outputshape)
        
    def forward(self,input_layer):
        return tf.reshape(input_layer,[-1]+self.outputshape)
    
    
class DenseLayer(Layer):
    
    def __init__(self,inputshape,nodes=1024,transfer_func=tf.nn.relu):
        super().__init__(inputshape,[nodes])
        self.nodes = nodes
        self.transfer_func = transfer_func
        
    def forward(self,input_layer):
        return tf.layers.dense(
                inputs=input_layer,
                units = self.nodes,
                activation = self.transfer_func)


class DropoutLayer(Layer):
    
    def __init__(self,inputshape,rate,mode):
        super().__init__(inputshape,inputshape)
        self.rate = 0.5
        self.mode = mode
    
    def forward(self,input_layer):
        return tf.layers.dropout(
                inputs = input_layer,
                rate = self.rate,
                training = self.mode[0] == 'TRAIN' )
        