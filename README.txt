#############################################################################
###############		README		#####################################
#############################################################################
KAGGLE PROJECT APPLIED MACHINE LEARNING
Data Science Bowl 2017
Rick Bruins en Stefan Haan
#############################################################################
PART 1: DATA PREPROCESSING
#############################################################################
All the code can be run from the main.py file. This file includes several
classess in order to create project related objects. The DataFile object
automatically loads image files located in ./data/train/ and ./data/test/
Creating a DataFile object requires to specify N and size, which are the
number of images to load and the image size respectively.
#############################################################################
PART 2: DATA AUGMENTATION
#############################################################################
Augmentation can be done by running the augment_train method off a DataFile
object, this requires you to specify a factor. This is the volume of the
images. A volume of two means multiplication of the data volume by 2.
#############################################################################
PART 3: CONVOLUTIONAL NEURAL NETWORK
#############################################################################
The network class allows you to create network object. Specify a data object
as input. Important methods in designing the network are specified and
commented in the main.py file. The network we used is commented out on the
bottom of the main.py file.
#############################################################################