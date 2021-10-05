import matplotlib.pyplot as plt
import numpy as np
from pix2pix import Pix2Pix
from test_data import Dataset_test
from load_testimage import *
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)
KTF.set_session(sess)


train_data = Dataset_test('test/') 
train_images = Dataset_test.load(train_data)

model = Pix2Pix( )
model_generator = model.build_generator()

# pretrained_weights.hdf5 can be downloaded from the link on our GitHub project page
model_generator.load_weights('generator.h5')

letter, number = model_generator.predict(train_images,batch_size = 16,verbose=1)

saveResult("result_draw/",letter)
saveResult("result_shuzi/",number)
