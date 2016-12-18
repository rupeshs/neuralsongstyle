import tensorflow as tf
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import stderr

CONTENT_FILENAME = "F:/Om/Projects/AudioStyleTransfer/neural-style-audio-tf/inputs/fade.mp3"
STYLE_FILENAME = "F:/Om/Projects/AudioStyleTransfer/neural-style-audio-tf/inputs/dont.mp3"
RESULT_FILENAME="F:/Om/Projects/AudioStyleTransfer/neural-style-audio-tf/outputs/outdont.wav"

# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs
    
a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)
r_style, fs = read_audio_spectum(RESULT_FILENAME)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Alan Walker Fade')
plt.imshow(a_content[:400,:])
plt.subplot(1,3,2)
plt.title('Chainsmokers Dont Let Me Down(style) ')
plt.imshow(a_style[:400,:])
plt.subplot(1,3,3)
plt.title('Result')
plt.imshow(r_style[:400,:])
plt.show()