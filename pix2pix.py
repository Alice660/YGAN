from __future__ import print_function, division
import scipy

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from data_set import Dataset
from data_set1 import Dataset1
from test_data import Dataset_test
from load_image import combine_images
import skimage.io as io
import cv2

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)
KTF.set_session(sess)

train_data = Dataset('train/',
                     'trainlabel_shuzi/') 
speckle_train, num = Dataset.load(train_data) 

train_data_1 = Dataset1('trainlabel_draw/')
eng = Dataset1.load(train_data_1)

val = Dataset_test('val_test/')
val_image = Dataset_test.load(val)

objects_num = 2

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.g_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.g_channels)
        self.d_input = (self.img_rows, self.img_cols, objects_num)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)  
        self.disc_patch = (patch, patch, 1)  

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images

        speckle = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A0,fake_A1 = self.generator(speckle)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A0,fake_A1])

        self.combined = Model(inputs=speckle, outputs=[valid, fake_A0, fake_A1])
        self.combined.compile(loss=['mse','binary_crossentropy','binary_crossentropy'],
                              loss_weights=[1, 100,100],
                              optimizer=optimizer)  #
        self.combined.summary()
# one input, two output
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True,dropout_rate=0):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',kernel_initializer = 'he_normal')(layer_input)
            
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',kernel_initializer='he_normal')(u)
            
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)  

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False) 
        d2 = conv2d(d1, self.gf*2)         
        d3 = conv2d(d2, self.gf*4,dropout_rate=0.3)         
        d4 = conv2d(d3, self.gf*8)         
        d5 = conv2d(d4, self.gf*8,dropout_rate=0.3)         
        d6 = conv2d(d5, self.gf*8)         
        d7 = conv2d(d6, self.gf*8)         

        # Upsampling,decoding eng
        u1 = deconv2d(d7, d6, self.gf*8)   
        u2 = deconv2d(u1, d5, self.gf*8)   
        
        u3 = deconv2d(u2, d4, self.gf*8)   
        u4 = deconv2d(u3, d3, self.gf*4)   # 
        
        u5 = deconv2d(u4, d2, self.gf*2)   # 
        u6 = deconv2d(u5, d1, self.gf)     # 

        u7 = UpSampling2D(size=2)(u6)      # 
        
        # Upsampling, decoding num
        n1 = deconv2d(d7, d6, self.gf*8)   # 
        n2 = deconv2d(n1, d5, self.gf*8)   # 
       
        n3 = deconv2d(n2, d4, self.gf*8)   # 
        n4 = deconv2d(n3, d3, self.gf*4)   # 
        
        n5 = deconv2d(n4, d2, self.gf*2)   # 
        n6 = deconv2d(n5, d1, self.gf)     # 

        n7 = UpSampling2D(size=2)(n6)      # 

        output_img1 = Conv2D(1, kernel_size=5, strides=1, padding='same',activation='sigmoid')(u7)

        output_img2 = Conv2D(1, kernel_size=5, strides=1, padding='same',activation='sigmoid')(n7)

        return Model(inputs = d0, outputs = [output_img1,output_img2])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, strides = 1,bn=True, dropout_rate=0):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = Dropout(0.3)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, f_size = 5, bn=False)  # 
        d2 = d_layer(d1, self.df*2,strides = 2)         # 
        d3 = d_layer(d2, self.df*4,strides = 2)         # 
        d4 = d_layer(d3, self.df*8,strides = 2)         # 
        d5 = d_layer(d4, self.df*8,strides = 2)         # 

        validity = Conv2D(1, kernel_size=5,  strides=1, padding='same')(d5)  # 

        return Model([img_A,img_B] ,validity)


    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)  # 
        fake = np.zeros((batch_size,) + self.disc_patch)
        loss_g = []
        for epoch in range(epochs):
            print("Epoch is", epoch)
            print("Number of batches", int(speckle_train.shape[0]/batch_size))
            for index in range(int(speckle_train.shape[0]/batch_size)):  #
                speckle_batch = speckle_train[index*batch_size:(index+1)*batch_size]  # 
                eng_batch = eng[index*batch_size:(index+1)*batch_size]
                num_batch = num[index*batch_size:(index+1)*batch_size]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A0,fake_A1 = self.generator.predict(speckle_batch)
                

                if index % 50 == 0:
                    val_draw,val_shuzi = self.generator.predict(val_image)

                    letter,number = combine_images(fake_A0,fake_A1)  # 
                    io.imsave(os.path.join('img_shuzi/',str(epoch) + '_' + str(index) +'.jpg'),number)
                    io.imsave(os.path.join('img_draw/',str(epoch) + '_' + str(index) +'.jpg'),letter)

                    draw,shuzi = combine_images(val_draw,val_shuzi)
                    io.imsave(os.path.join('img_val_shuzi/',str(epoch) + '_' + str(index) +'.jpg'),shuzi)
                    io.imsave(os.path.join('img_val_draw/',str(epoch) + '_' + str(index) +'.jpg'),draw)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([eng_batch,num_batch], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A0,fake_A1], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch(speckle_batch, [valid,eng_batch,num_batch])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,
                                                                        index,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if index % sample_interval == 0:
                    self.generator.save('generator.h5')
                    self.discriminator.save('discriminator.h5')
                    loss_g.append(g_loss[0])
            
        loss_g = np.array(loss_g)
        np.savetxt('filename.csv',loss_g)        



if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=18, batch_size=8, sample_interval=50)
