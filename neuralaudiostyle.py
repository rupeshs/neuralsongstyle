import torch
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import stderr

# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectrum(filename):
    x, fs = librosa.load(filename)
    print("sampling rate :",fs)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs

def train_style(a_content, a_style):
    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]

    print("audio style train")
    N_FILTERS = 4096

    a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
    a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
    kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std

    g = torch.nn.Sequential(
        torch.nn.Conv2d(1, N_FILTERS, (1, 11)),
        torch.nn.ReLU()
    )

    content_features = g(torch.tensor(a_content_tf, dtype=torch.float32)).detach().numpy()
    style_features = g(torch.tensor(a_style_tf, dtype=torch.float32)).detach().numpy()

    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES

    return content_features, style_gram

def optimize(content_features, style_gram, N_SAMPLES, N_CHANNELS, N_FILTERS, output_path, fs):
    ALPHA= 1e-2
    learning_rate= 1e-3
    iterations = 100
    print("optimise")
    result = None

    g = torch.nn.Sequential(
        torch.nn.Conv2d(1, N_FILTERS, (1, 11)),
        torch.nn.ReLU()
    )

    x = torch.tensor(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, requires_grad=True)
    optimizer = torch.optim.LBFGS([x], lr=learning_rate, max_iter=iterations)

    def closure():
        optimizer.zero_grad()
        net = g(x)
        content_loss = ALPHA * 2 * torch.nn.functional.mse_loss(net, torch.tensor(content_features, dtype=torch.float32))
        style_loss = 0

        _, height, width, number = net.shape
        size = height * width * number
        feats = net.view(-1, number)
        gram = torch.matmul(feats.t(), feats) / N_SAMPLES
        style_loss = 2 * torch.nn.functional.mse_loss(gram, torch.tensor(style_gram, dtype=torch.float32))

        loss = content_loss + style_loss
        loss.backward()
        return loss

    optimizer.step(closure)
    result = x.detach().numpy()

    a = np.zeros_like(content_features[0,0].T)
    a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    librosa.output.write_wav(output_path, x, fs)

def perform_style_transfer(content, style, output):
    a_content, fs = read_audio_spectrum(content)
    a_style, _ = read_audio_spectrum(style)

    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    N_FILTERS = 4096

    content_features, style_gram = train_style(a_content, a_style)
    optimize(content_features, style_gram, N_SAMPLES, N_CHANNELS, N_FILTERS, output, fs)
