import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from natsort import natsorted

import tensorflow_addons as tfa

import cv2
import shutil
import glob

import tensorflow as tf
from tensorflow.keras import layers, models, Input

import sklearn

image_shape = (28,28,1)

img_dim = image_shape[0]


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=5, strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=5, strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*4),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Reshape(target_shape=(7, 7, 4)),

                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.UpSampling2D(),

                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.UpSampling2D(),

                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    # def decode(self, z, apply_sigmoid=False):
    def decode(self, z, apply_sigmoid=True):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

latent_dim = 32
model = CVAE(latent_dim)

def compute_loss(x, c, alpha, eps = 1e-4):
    z_mean, z_log_var = model.encode(x)
    z = model.reparameterize(z_mean, z_log_var)
    reconstruction = model.decode(z)

    latent_features = np.array(z)

    dist = tf.math.reduce_sum((z - c) ** 2, axis=1)

    # print("dist:", dist)

    euclidean_norm = tf.math.sqrt(dist)

    dist_loss = tf.math.reduce_mean(euclidean_norm)

    # print("dist_loss:", dist_loss)

    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
        )
    )
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    vae_loss = reconstruction_loss + kl_loss

    loss = alpha * vae_loss + dist_loss

    return loss, euclidean_norm

def dataset_collection_func(normal_class):

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    one_class_idx = np.where(train_labels == normal_class)
    train_images = train_images[one_class_idx]

    test_labels[test_labels!=normal_class] = 11
    test_labels[test_labels==normal_class] = 0
    test_labels[test_labels==11] = 1
    
    return train_images, test_images, test_labels

train_images, test_images, test_labels = dataset_collection_func(normal_class = 8)

learning_rate = 1e-3
# optimizer = tf.keras.optimizers.Adam(learning_rate)
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

checkpoint_dir = './Deep_SVDD_VAE_checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

checkpoint.restore(ckpt_manager.latest_checkpoint) # restore the lastest checkpoints

def my_metrics(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for prediction, y in zip(y_pred, y_true):

        if(prediction == y):
            if(prediction == 1):
                TP += 1
            else:
                TN += 1
        else:
            if(prediction == 1):
                FP += 1
            else:
                FN += 1

    precision = TP/(TP+FP+1.0e-4)

    recall = TP/(TP+FN+1.0e-4)

    f_measure = (2. * precision * recall)/(precision + recall + 1.0e-4)

    accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)

    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("f_measure:", f_measure)
    # print("accuracy:", accuracy)

    return np.array([TP, TN, FP, FN, precision, recall, f_measure, accuracy])

def train_step(inputs_image, c, optimizer, alpha):
    # print("training......")

    with tf.GradientTape() as tape:
        loss, _ = compute_loss(inputs_image, c, alpha)

        # print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.array(loss), optimizer

def get_R(dist, nu):
    return np.quantile(dist, 1 - nu)

def get_R_based_on_c(train_images, nu, c, eps = 1e-4):

    latent_features_list = []

    for index in range(len(train_images)):
        image = train_images[index]
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        image = image.astype(np.float32)
        image = image / 255.

        mean, logvar = model.encode(image)
        latent_features = model.reparameterize(mean, logvar)
        latent_features = np.array(latent_features)

        latent_features_list.append(latent_features[0])

    latent_features_list = np.array(latent_features_list)

    dist = np.sum((latent_features_list - c) ** 2, axis=1)

    euclidean_norm = np.sqrt(dist)

    R = get_R(euclidean_norm, nu)

    # R = tf.Variable(R, dtype=tf.float32, trainable=False)

    return R

def get_c_and_R(train_images, nu, eps = 1e-4):

    N = 0
    c_sum = np.zeros([latent_dim])

    latent_features_list = []

    for index in range(len(train_images)):
        image = train_images[index]
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        image = image.astype(np.float32)
        image = image / 255.

        mean, logvar = model.encode(image)
        latent_features = model.reparameterize(mean, logvar)
        latent_features = np.array(latent_features)
        c_sum += latent_features[0]

        latent_features_list.append(latent_features[0])

        N += 1

    latent_features_list = np.array(latent_features_list)

    # print("N:",N)
    
    c = c_sum / N

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    dist = np.sum((latent_features_list - c) ** 2, axis=1)

    euclidean_norm = np.sqrt(dist)

    R = get_R(euclidean_norm, nu)

    c = tf.Variable(c, dtype=tf.float32, trainable=False)
    # R = tf.Variable(R, dtype=tf.float32, trainable=False)

    return c, R

def customized_R_evaluation(train_images, batch_images, batch_labels, nu, c):

    if(c is None):
        c, R = get_c_and_R(train_images, nu)
    else:
        R = get_R_based_on_c(train_images, nu, c)

    print("R:", R)

    predictions = []

    scores_list = []
    dist_list = []

    for i in range(len(batch_labels)):

        img_array = batch_images[i]

        img_array = np.expand_dims(img_array, axis=0)

        img_array = np.expand_dims(img_array, axis=-1)

        img_array = img_array.astype(np.float32)

        img_array = img_array / 255.

        mean, logvar = model.encode(img_array)
        z = model.reparameterize(mean, logvar)

        dist = np.sum((z[0] - c) ** 2)
        dist = np.sqrt(dist)

        dist_list.append(dist)

        scores = R - dist

        scores_list.append(scores)

        if(scores >= 0):
            predictions.append(0)
        else:
            predictions.append(1)

    labels = np.array(batch_labels)
    predictions = np.array(predictions)

    metric_results = my_metrics(labels, predictions)

    scores_list = np.array(scores_list)
    dist_list = np.array(dist_list)

    auc_roc = sklearn.metrics.roc_auc_score(labels, dist_list)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

    return metric_results, auc_roc

def train(train_images, test_images, test_labels, epochs, BATCH_SIZE, nu, alpha):

    global learning_rate
    global optimizer

    best_auc_roc = 0

    for epoch in range(epochs):
        start = time.time()

        idx = np.random.permutation(len(train_images))
        train_images = train_images[idx]

        print("train epoch = ",epoch)
        c, _ = get_c_and_R(train_images, nu)
        for index in range(0, len(train_images)-BATCH_SIZE, BATCH_SIZE):
            label_batch = []

            for i in range(BATCH_SIZE):

                img_array = train_images[index+i]
                img_array = np.expand_dims(img_array, axis=-1)

                img_array = img_array.astype(np.float32)
                img_array = img_array / 255.

                # data augmentation
                img_array = tf.keras.preprocessing.image.random_rotation(img_array, 0.2)
                img_array = tf.keras.preprocessing.image.random_shift(img_array, 0.1, 0.1)
                img_array = tf.keras.preprocessing.image.random_shear(img_array, 0.1)
                img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.7,1))

                img_array = np.array(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                if(i == 0):
                    image_batch = img_array
                else:
                    image_batch = np.concatenate((image_batch, img_array), axis=0)

            loss, optimizer = train_step(image_batch, c, optimizer, alpha)

            # print("training Loss: ", loss)

        if(epoch % 10 == 0):

            idx_val = np.random.permutation(len(test_labels))
            test_images, test_labels = test_images[idx_val], test_labels[idx_val]

            metric_results, auc_roc = customized_R_evaluation(train_images, test_images, test_labels, nu, c)

            if(auc_roc > best_auc_roc):
                best_auc_roc = auc_roc

                f = open("best_auc_roc_deep_svdd_vae.txt", "w")
                f.write(str(best_auc_roc))
                f.close()

        if(epoch == 200):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)

        if(epoch == 250):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    print("saveing model")
    ckpt_manager.save()

    np.save("c_deep_svdd_vae.npy", np.array(c)) # save c

epochs = 300 + 1
BATCH_SIZE = 200
nu = 0.05 # it should be a fine-tuned hyper-parameter
alpha = 50
train(train_images, test_images, test_labels, epochs, BATCH_SIZE, nu, alpha)
