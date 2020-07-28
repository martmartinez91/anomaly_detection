import networkx as nx
import pickle
#import multiprocessing
from multiprocessing import Pool
from itertools import product
import copy
import sys
import random
import math
import pandas as pd
import numpy as np
# import tensorflow as tf
import itertools
from collections import Counter
#import matplotlib.pyplot as plt
#plt.use('Agg')
from pathlib import Path
#from datetime import datetimeimport networkx as nx
import pickle
#import multiprocessing
from multiprocessing import Pool
from itertools import product
import copy
import sys
import random
import math
import pandas as pd
import numpy as np
# import tensorflow as tf
import itertools
from collections import Counter
#import matplotlib.pyplot as plt
#plt.use('Agg')
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import sys
import pandas as pd
print(sys.version) # python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
print(torch.__version__) # 1.0.1

#%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

#import tensorflow as tf
#from tensorflow.python.keras.layers import Input, Dense


from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam




def setup():

    #STEP 1

    dataset_path = '1Week_Input.csv'
    #dataset_path = '12Hrs_Input.csv'

    ledger = pd.read_csv(dataset_path, dtype={'time': float, 'dev1': str, 'dev2': str, 'connection': str})
    current_time = datetime.now()
    current_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")

    print(ledger)

    start_time_ledger = ledger['time'].iloc[0] # Start time of the timestamp in the ledger
    finish_time_ledger = ledger['time'].iloc[-1] # Finish time of the timestamp in the ledger

    start_time = start_time_ledger
    finish_time = finish_time_ledger

    print("start time : " + str(start_time) + ' - finish time: ' + str(finish_time))

    init_ledger = ledger.loc[(ledger['time'] >= start_time) & (ledger['time'] < finish_time)]

    print(init_ledger)

    init_graph = nx.from_pandas_edgelist(init_ledger, 'dev1', 'dev2', create_using=nx.MultiGraph())
    init_nodes = list(init_graph.nodes)

    init_nodes
    len(init_nodes)

    ############################################################################################################################################

    nrows = 12*14 + 1
    #nrows = 12
    ncols = len(init_nodes)

    column_names = init_nodes
    # print(column_names)
    # row_names = init_nodes
    #row_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12','t13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24',

    row_names = []
    for i in range(0,nrows):
        row_names.append('t'+ str(i+1))

    # print(row_names)


    att_period = 3600
    main_attacker_selection = init_nodes[:math.ceil(len(init_nodes) / 2)]

    # example  = ['7']


    lista_de_listas = []

    image_list = []

    # for main_attacker in init_nodes:

    for main_attacker in init_nodes:

        # print(main_attacker)

        image_list = []

        for time in np.arange(start_time, finish_time + 1, att_period):
            # column_names = ['a', 'b', 'c']
            # row_names    = ['1', '2', '3']

            # matrix = np.reshape((1, 2, 3, 4, 5, 6, 7, 8, 9), (3, 3))
            # df = pd.DataFrame(matrix, columns=column_names, index=row_names)
            # df

            # print(time)
            left_att = init_ledger.loc[(init_ledger['dev1'] == main_attacker) & (init_ledger['time'] >= time) &
                                       (init_ledger['time'] < time + att_period)]

            one_side = left_att.dev2.unique().tolist()
            # print(one_side)
            right_att = init_ledger.loc[(init_ledger['dev2'] == main_attacker) & (init_ledger['time'] >= time) &
                                        (init_ledger['time'] < time + att_period)]
            other_side = right_att.dev1.unique().tolist()
            # print(other_side)

            full_list = one_side + other_side
            # print(full_list)

            image_list.append(full_list)

        print("Attacker is " + str(main_attacker) + "  ")
        #print(image_list)
        # HASTA ESTE PUNTO SE TIENE TODA LA INFO DE LAS 12 IMAGENES DE UN ATTACKER


        arr_final = []

        #matrix = np.zeros(564).reshape((12, 47))
        matrix = np.zeros(ncols*nrows).reshape((nrows, ncols))
        df = pd.DataFrame(matrix, columns=column_names, index=row_names)

        i = 0

        for linea in image_list:

            #print(linea)
            # print(init_nodes)

            # matrix = np.zeros(2209).reshape((47, 47))
            # df = pd.DataFrame(matrix, columns=column_names, index=row_names)
            #print("Linea N#:  " + str(i+1))
            for column in linea:
                #print(column)
                df.at[row_names[i], column] = 1

                arr = df.to_numpy()
                arr.shape

                # arr_final.append(arr)

            i = i + 1
            # plt.imshow(df)
            # plt.xticks(range(ncols), column_names)
            # plt.yticks(range(nrows), row_names)
            # plt.show()

        arr_final = arr
        b = torch.FloatTensor(arr_final)
        #print(b)
        lista_de_listas.append(b)


    for i in range(0, ncols):
        print(lista_de_listas[i].size())

        # print(lista_de_listas[40])

    #  GENERATE A RANDOM TENSOR TO SIMULATE A FAKE DATA FROM GENERATOR
    #  ADD THE FAKE INFORMATION TO THE REAL TENSORS
    #  ONCE IT WORKS WITH ONE PROCEED TO DO THE SAME FOR 10 CASES (10 GEN)

    random.seed(0)



    # GENERATE A RANDOM ATTACK
    #attacker_list = []

    #for j in range(0, 5):
        # GENERATE A 12X47 TENSOR FAKE DATA
    #    sample_attacker = np.random.randint(2, size=(12, 47))
    #    print(sample_attacker)
    #    attacker_list.append(sample_attacker)
        # print(sample_attacker.size())


    #MOVING ALL THE ORIGINAL TENSOR TO MOD REAL SINCE THIS WILL BE PADDED WITH FAKE DATA
    #mod_Real = lista_de_listas

    #Check that the tensors have now a size 12x52
    for i in range(0,ncols):
        print(lista_de_listas[i].size())


    return lista_de_listas




    for i in range(0, 47):

        for j in range(0, 5):
            temp = attacker_list[j]

            sel_column = torch.from_numpy(temp[:, i])
            sel_column = sel_column.unsqueeze(1)
            print(sel_column)
            print(sel_column.size())

            print(mod_Real[i].size())
            x = torch.cat((mod_Real[i], sel_column.float()), 1)
            print(x)
            print(x.size())
            mod_Real[i] = x


    #Check that the tensors have now a size 12x52
    for i in range(0,47):
        print(mod_Real[i].size())


    ########################################################################################################
    # MAKE ALL THE FAKE DATA TO HAVE THE SAME TENSORS 12 X 52
    # attacker_list , has 5 fake copies
    # idea is to add one column without anything and pad the remaining column

    mod_Fake = attacker_list
    print(mod_Fake)

    zeroes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(0, 5):

        for j in range(0, 5):

            if i == j:
                print(i)
                result = np.column_stack((mod_Fake[i], zeroes))
                # an_array = np.insert(mod_Fake[i],48,zeroes, axis =1)
                # print(result)
                mod_Fake[i] = result

            else:
                #result = np.column_stack((mod_Fake[i], attacker_list[j]))
                result = np.column_stack((mod_Fake[i], zeroes))
                mod_Fake[i] = result

    for i in range(0, 5):
        mod_Fake[i] = torch.FloatTensor(mod_Fake[i])
        print(mod_Fake[i])


    for i in range(0, 5):
        print(mod_Fake[i].size())


    return lista_de_listas




##############################################################################################################################################

img_rows = 169
#img_rows = 12
img_cols = 104
#img_cols = 57
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100


def build_generator(img_shape, z_dim):

    model = Sequential()

    # Fully connected layer
    model.add(Dense(128, input_dim=z_dim))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with tanh activation
    model.add(Dense(169 * 104 * 1, activation='tanh'))

    # Reshape the Generator output to image dimensions
    model.add(Reshape(img_shape))

    return model


def build_discriminator(img_shape):

    model = Sequential()

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Fully connected layer
    model.add(Dense(128))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model



# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(img_shape, z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []



def column(matrix, i):
    return [row[i] for row in matrix]




def train(iterations, batch_size, sample_interval):

# BATCH SIZE = 47


    mod_Real = setup()
    # Load the MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    print(mod_Real)
    X_train = [t.numpy() for t in mod_Real]
    X_train = np.expand_dims(X_train, axis=3)

    #print(mnist.load_data())

    # Rescale [0, 255] grayscale pixel values to [-1, 1]

    # convert the info on image to -1 to +1 range
    #X_train = X_train / 127.5 - 1.0
    # convert the shape from 60000,28,28 to 60000,28,28,1
    #X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones (128 labels of real all in one column)
    # Noflabels  = 47
    #real = np.ones((47, 1))
    real = np.ones((94, 1))

    # Labels for fake images: all zeros (128 labels of fake all in one column)
    # Noflabels = 5
    #fake = np.zeros((5, 1))
    fake = np.zeros((10, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images
        # idx selects the index of the images to use for training
        # x_train.shape[0] is the options from 0 to 59999 to select
        #idx = np.random.randint(0, X_train.shape[0], batch_size)
        # imgs has a tuple with 128,28,28,1
        # imgs has a tuple with 47,12,47,1
        #imgs = X_train[idx]
        imgs = X_train

        # Generate a batch of fake images
        # z will have 128 rows, 100 columns and random input using normal distribution
        # batch_size = 5

        #z = np.random.normal(0, 1, (5, 100))
        z = np.random.normal(0, 1, (10, 100))
        # gen_imgs will get 128,28,28,1 shape
        # gen_mgs will get 5,12,47,1
        gen_imgs = generator.predict(z)
        gen_imgs[gen_imgs < 0] = 0
        gen_imgs[gen_imgs > 0] = 1
        # need to turn neg values to 0s and positive to 1s



        # Adjust both the imgs and gen_imgs
        # padding with zeroes so 12,47 becomes 12,52
        #new_images =  np.zeros(shape=(47,12,52))
        new_images = np.zeros(shape=(94, 169, 104))
        new_images = np.expand_dims(new_images, axis=3)

        # One piece of code to add column to imgs
        # RANGE: 47
        for i in range(0, 94):

            # RANGE: 12
            for x in range(0,169):

                # RANGE:47
                for y in range(0,94):

                        new_images[i][x][y] = imgs[i][x][y]


            # FOR LOOP FROM O TO 5
            for j in range(0, 10):
                temp = gen_imgs[j]

                #sel_column must have the ith column of the jth fake image
                sel_column = column(temp , i)
                #print(sel_column)
                sel_column = np.expand_dims(sel_column, axis=2)
                #print(sel_column.size())


                # FOR LOOP FROM O TO 12
                for k in range(0,169):
                    # FOR LOOP FROM J+47
                    new_images[i][k][j+94] = sel_column[k]

                #print(imgs[i].size())
                #x = torch.cat((imgs[i], sel_column.float()), 1)
                #x = np.hstack((imgs[i],sel_column))
                #new_images[i][][j+47] = sel_column
                #print(x)
                #print(x.size())
                #imgs[i] = x


        # code for padding the fake imgs
        #zeroes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        #for i in range(0, 5):

#            for j in range(0, 5):

#                if i == j:
#                    print(i)
#                    result = np.column_stack((gen_imgs[i], zeroes))
                    # an_array = np.insert(mod_Fake[i],48,zeroes, axis =1)
                    # print(result)
#                    gen_imgs[i] = result

#                else:
                    # result = np.column_stack((mod_Fake[i], attacker_list[j]))
                    #result = np.column_stack((gen_imgs[i], zeroes))
                    #gen_imgs[i] = result



        # Train Discriminator
        # check the train result for real images and real labels
        # a total of 47 images we need to adjust
        d_loss_real = discriminator.train_on_batch(new_images, real)
        # check the train result for gen fake images and fake labels
        # a total of 5 images we need to adjust
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        # get the accuracy average
        #d_loss, accuracy = 0.5 * np.add(d_loss_real*(5/52), d_loss_fake*(47/52))
        #d_loss = 0.5*(d_loss_fake[0]*(47/52) + d_loss_real[0]*(5/52))
        #accuracy = 0.5 * (d_loss_fake[1] * (47 / 52) + d_loss_real[1] * (5 / 52))
        d_loss = 0.5 * (d_loss_fake[0] * (94 / 104) + d_loss_real[0] * (10 / 104))
        accuracy = 0.5 * (d_loss_fake[1] * (94 / 104) + d_loss_real[1] * (10 / 104))
        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        # BATCH SIZE = 5
        z = np.random.normal(0, 1, (10, 100))
        gen_imgs = generator.predict(z)
        gen_imgs[gen_imgs < 0] = 0
        gen_imgs[gen_imgs > 0] = 1

        # Train Generator
        # train on the current batch of fake images
        # real is a list of 5 labels
        #labels_real = [1,1,1,1,1]
        labels_real = np.ones((10, 1))
        g_loss = gan.train_on_batch(z, labels_real)

        if (iteration + 1) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator)




def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1



# Set hyperparameters
iterations = 1000
batch_size = 5
sample_interval = 50

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)