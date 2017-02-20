import model
import theano_funcs
import utils
import pickle

from iter_funcs import get_batch_idx

import numpy as np
from lasagne.layers import get_all_param_values
from os.path import join
import cv2





def train_autoencoder(X_train):
    print('building model')
    layers = model.build_model()

    max_epochs = 30
    batch_size =72
    weightsfile = join('weights', 'weights_train_val.pickle')

    print('compiling theano functions for training')
    print('  encoder/decoder')
    encoder_decoder_update = theano_funcs.create_encoder_decoder_func(
        layers, apply_updates=True)
    print('  discriminator')
    discriminator1_update = theano_funcs.create_discriminator1_func(
        layers, apply_updates=True)
    discriminator2_update = theano_funcs.create_discriminator2_func(
        layers, apply_updates=True)
    print('  generator')
    generator1_update = theano_funcs.create_generator1_func(
        layers, apply_updates=True)
    generator2_update = theano_funcs.create_generator2_func(
        layers, apply_updates=True)
    final_encoder, final_decoder = theano_funcs.show_result(layers)





    try:
        for epoch in range(1, max_epochs + 1):
            print('epoch %d' % (epoch))

            # compute loss on training data and apply gradient updates
            train_reconstruction_losses = []
            train_discriminative1_losses = []
            train_generative1_losses = []
            train_discriminative2_losses = []
            train_generative2_losses = []

            for train_idx in get_batch_idx(X_train.shape[0], batch_size):
                X_train_batch = X_train[train_idx]
                # 1.) update the encoder/decoder to min. reconstruction loss
                train_batch_reconstruction_loss =\
                    encoder_decoder_update(X_train_batch)

                # sample from p(z)
                pz_train_batch = np.random.normal(
                    -1, 1,
                    size=(X_train_batch.shape[0], 3)).astype(
                        np.float32)
                py_train_batch=[]
                for i in range(X_train_batch.shape[0]):
                    a=[]
                    r=np.random.random()
                    a.append(r)
                    a.append(1-r)
                    py_train_batch.append(a)
                py_train_batch=np.array(py_train_batch).astype(np.float32)
                train_batch_discriminative1_loss = \
                    discriminator1_update(X_train_batch, py_train_batch)

                # 2.) update discriminator to separate q(z|x) from p(z)
                train_batch_discriminative2_loss =\
                    discriminator2_update(X_train_batch, pz_train_batch)

                # 3.)  update generator to output q(z|x) that mimic p(z)
                train_batch_generative1_loss = generator1_update(X_train_batch)
                train_batch_generative2_loss = generator2_update(X_train_batch)

                train_reconstruction_losses.append(
                    train_batch_reconstruction_loss)
                train_discriminative1_losses.append(
                    train_batch_discriminative1_loss)
                train_generative1_losses.append(
                    train_batch_generative1_loss)
                train_discriminative2_losses.append(
                    train_batch_discriminative2_loss)
                train_generative2_losses.append(
                    train_batch_generative2_loss)

            # average over minibatches
            train_reconstruction_losses_mean = np.mean(
                train_reconstruction_losses)
            train_discriminative1_losses_mean = np.mean(
                train_discriminative1_losses)
            train_discriminative2_losses_mean = np.mean(
                train_discriminative2_losses)

            train_generative1_losses_mean = np.mean(
                train_generative1_losses)
            train_generative2_losses_mean = np.mean(
                train_generative2_losses)

            print('  train: rec = %.6f, dis1 = %.6f,dis2 = %.6f, gen1 = %.6f,gen2 = %.6f' % (
                train_reconstruction_losses_mean,
                train_discriminative1_losses_mean,
                train_discriminative2_losses_mean,
                train_generative1_losses_mean,
                train_generative2_losses_mean,
            ))

        data_encoder = final_encoder(X_train)
        f = open('data_encoder.pkl', 'wb')
        pickle.dump(data_encoder, f)
        f.close()

        data_decoder = final_decoder(X_train)
        f = open('data_decoder.pkl', 'wb')
        pickle.dump(data_decoder, f)
        f.close()
        print 'data saved successfuly!'
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    img = cv2.imread('lena.jpg')
    X_train = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X_train.append(img[i][j])

    X_train = np.array(X_train).astype(np.float32)
    X_train/=255
    train_autoencoder(X_train)


