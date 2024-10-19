import tensorflow as tf
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import stderr
import argparse
import logging

parser = argparse.ArgumentParser(description='Neurals style transfer for songs')
parser.add_argument('--content',help='Content audio path',required=True)
parser.add_argument('--style',help='Style audio path',required=True)
parser.add_argument('--out',help='Styled audio path',required=True)
args = parser.parse_args()

logging.basicConfig(filename='neuralaudiostyle.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

N_FFT = 2048
def read_audio_spectrum(filename):
    try:
        x, fs = librosa.load(filename, sr=None)
        logging.info(f"Loaded {filename} with sampling rate {fs}")
        S = librosa.stft(x, N_FFT)
        p = np.angle(S)
        
        S = np.log1p(np.abs(S[:,:430]))  
        return S, fs
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        raise

def train_style(a_content, a_style):
    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]

    logging.info("Training style")
    N_FILTERS = 4096

    a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
    a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

    std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
    kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std
        
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        
        net = tf.nn.relu(conv)

        content_features = net.eval(feed_dict={x: a_content_tf})
        style_features = net.eval(feed_dict={x: a_style_tf})
        
        features = np.reshape(style_features, (-1, N_FILTERS))
        style_gram = np.matmul(features.T, features) / N_SAMPLES

    return content_features, style_gram, kernel

def optimize(content_features, style_gram, kernel, N_SAMPLES, N_CHANNELS):
    ALPHA= 1e-2
    learning_rate= 1e-3
    iterations = 100
    logging.info("Starting optimization")
    result = None
    with tf.Graph().as_default():
        x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        
        net = tf.nn.relu(conv)

        content_loss = ALPHA * 2 * tf.nn.l2_loss(
                net - content_features)

        style_loss = 0

        _, height, width, number = map(lambda i: i.value, net.get_shape())

        size = height * width * number
        feats = tf.reshape(net, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

        loss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
              loss, method='L-BFGS-B', options={'maxiter': 300})
            
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
           
            logging.info('Started optimization.')
            opt.minimize(sess)
        
            logging.info(f'Final loss: {loss.eval()}')
            result = x.eval()
            
    return result

def perform_style_transfer(content, style, output):
    a_content, fs = read_audio_spectrum(content)
    a_style, fs = read_audio_spectrum(style)

    content_features, style_gram, kernel = train_style(a_content, a_style)
    result = optimize(content_features, style_gram, kernel, a_content.shape[1], a_content.shape[0])

    a = np.zeros_like(a_content)
    a[:a_content.shape[0],:] = np.exp(result[0,0].T) - 1

    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    librosa.output.write_wav(output, x, fs)
    logging.info(f"Style transfer complete. Output saved to {output}")

if __name__ == "__main__":
    perform_style_transfer(args.content, args.style, args.out)
