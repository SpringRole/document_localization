import os
import numpy as np
import cv2, random
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.models import load_model
from numpy import load
from numpy import savez_compressed
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default = None, help="path to *specific* model checkpoint to load")
parser.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
parser.add_argument("-d", "--data_path", type=str, help="path of dataset")
parser.add_argument("-cn", "--last_checkpoint_npz", type=str, default=None,
                    help="last saved npz file from which continuation is required")
parser.add_argument("-r", "--root_path", type=str, help="root path to save your checkpoint")
parser.add_argument("-c", '--csv_path', type=str, help='location to save performance csv')
args = parser.parse_args()
model = args.model
start_epoch = args.start_epoch
data_path = args.data_path
last_checkpoint_npz = args.last_checkpoint_npz
root_path = args.root_path
csv_path = args.csv_path

if not os.path.exists(root_path):
	os.mkdir(root_path)

if not os.path.exists(csv_path):
	os.mkdir(csv_path)
	

df = pd.DataFrame(columns=['epoch', 'd1', 'd2', 'g_loss', 'avg'])
def load_data(req_shape=(256, 256, 3)):
    global data_path
    
    data  = os.listdir(data_path)
    
    data0 = load(data_path + data[0])
    X_images1, X_target1 = data0['arr_0'], data0['arr_1']
    
    data1 = load(data_path + data[1])
    X_images2, X_target2 = data1['arr_0'], data1['arr_1']
    
    data2 = load(data_path + data[2])
    X_images3, X_target3 = data2['arr_0'], data2['arr_1']
    
    data3 = load(data_path + data[3])
    X_images4, X_target4 = data3['arr_0'], data3['arr_1']
    
    
    X_images = np.concatenate([X_images1, X_images2, X_images3, X_images4])
    X_target = np.concatenate([X_target1, X_target2, X_target3, X_target4])
    
    return (X_images, X_target)
X_images, X_target = load_data()
########visualization########
X_images = (X_images - 127.5) / 127.5
X_target = (X_target - 127.5) / 127.5
"""**discriminator**"""
# define the discriminator model
def define_discriminator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model
"""**generator**"""
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    # print(g.shape, skip_in.shape)
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
# define the standalone generator model
def define_generator(image_shape=(256, 256, 3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False
    in_src = Input(shape=image_shape)  # input image
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])  # tanh values
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model
def generate_real_samples(X_images, X_target, n_samples, patch_shape):
    sample = np.random.randint(0, X_images.shape[0], n_samples)
    real_i = X_images[sample]
    real_t = X_target[sample]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [real_i, real_t], y
def generate_fake_samples(gen_model, samples, patch_shape):
    X = gen_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y
last_inserted = last_checkpoint_npz  # '/content/gdrive/My Drive/GAN/checkpoint/66858/satellite_images_0.8269082307815552_0.22247761487960815_2.68192720413208.npz'
def save_data(gen_loss_arrray, d_loss1, d_loss2, epoch, avg):
    global last_inserted, root_path, df
    root = root_path  # '/content/gdrive/My Drive/GAN/checkpoint'
    save_path = root_path + "/" + "{}/".format(epoch)  # '/content/gdrive/My Drive/GAN/checkpoint/{}/'.format(epoch)
    os.mkdir(save_path)
    df = df.append(
        {'epoch': epoch,
         'd1': d_loss1[-1],
         'd2': d_loss2[-1],
         'g_loss': gen_loss_arrray[-1],
         'avg': avg},
        ignore_index=True
    )
    df.to_csv(csv_path + '/' + 'performance.csv', index=False, encoding='utf-8')
    if len(os.listdir(root)) == 0:
        filename = save_path + 'satellite_images_{}_{}_{}.npz'.format(d_loss1[-1], d_loss2[-1], gen_loss_arrray[-1])
        last_inserted = filename
        plt.plot(gen_loss_arrray, label='generator')
        plt.plot(d_loss1, label='disc1')
        plt.plot(d_loss2, label='disc2')
        plt.legend()
        plt.savefig(save_path + '{}.png'.format(gen_loss_arrray[-1]))
        savez_compressed(filename, d_loss1, d_loss2, gen_loss_arrray)
    else:
        # load previous
        print(last_inserted)
        previous_npz = load(last_inserted)
        d1_prev_loss, d2_prev_loss, gen_prev_loss = previous_npz['arr_0'], previous_npz['arr_1'], previous_npz['arr_2']
        # merge
        current_gen_losses = np.concatenate([gen_prev_loss, gen_loss_arrray])
        current_d1_losses = np.concatenate([d1_prev_loss, d_loss1])
        current_d2_losses = np.concatenate([d2_prev_loss, d_loss2])
        # plot and save
        plt.plot(np.log10(current_gen_losses), label='generator')
        plt.plot(current_d1_losses, label='disc1')
        plt.plot(current_d2_losses, label='disc2')
        plt.legend()
        plt.savefig(save_path + '{}.png'.format(gen_loss_arrray[-1]))
        filename = save_path + 'satellite_images_{}_{}_{}.npz'.format(d_loss1[-1], d_loss2[-1], gen_loss_arrray[-1])
        last_inserted = filename
        savez_compressed(filename, current_d1_losses, current_d2_losses, current_gen_losses)
    plt.close()
def train(d_model, g_model, gan_model, X_images, X_target, n_epochs=300, n_batch=1):
    global root_path
    if start_epoch == 0:
    	prev_best = 0
    else:
    	prev_best = float(last_checkpoint_npz.split('_')[-1][:-4])
    n_patch = d_model.output_shape[1]
    trainA, trainB = X_images, X_target
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    gen_loss_arrray = []
    d1_loss_array = []
    d2_loss_array = []
    avg = -1
    st_ep = 500
    d_loss1 = d_loss2 = 0
    for i in range(start_epoch, n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(X_images, X_target, n_batch, n_patch)
        # only train discriminator on every 500 steps
        if i % st_ep == 0:
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator on every step
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        gen_loss_arrray.append(g_loss)
        d1_loss_array.append(d_loss1)
        d2_loss_array.append(d_loss2)
        if avg == -1:
            avg = g_loss
        else:
            avg = (avg + g_loss) / 2
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f] a[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss, avg))
        # summarize model performance
        if g_loss < prev_best:
            prev_best = g_loss
            save_data(gen_loss_arrray, d1_loss_array, d2_loss_array, i, avg)
            g_model.save(root_path + "/" + "{}/epoch_{}.h5".format(i, i))
            gen_loss_arrray = []
            d1_loss_array = []
            d2_loss_array = []
        if i % 3500 == 0:
            prev_best = g_loss
            save_data(gen_loss_arrray, d1_loss_array, d2_loss_array, i, avg)
            g_model.save(root_path + "/" + "{}/epoch_{}.h5".format(i, i))
            gen_loss_arrray = []
            d1_loss_array = []
            d2_loss_array = []
image_shape = (256, 256, 3)
# define the models
d_model = define_discriminator(image_shape)
if model is None:
    g_model = define_generator(image_shape)
else:
    print("[INFO] loading {}...".format(model))
    g_model = load_model(model)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, X_images, X_target)
g_model.save(root_path + '/' + 'gan_final_saved_model.h5')
