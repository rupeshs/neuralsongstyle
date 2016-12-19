# neuralsongstyle
Transferring the style from one song onto another using artificial intelligence.

This is based on the implementation [neural-style-audio](https://github.com/DmitryUlyanov/neural-style-audio-tf) by Dmitry and Vadim.

### Options
 - --content ,Content audio path
 - --style, Style audio path
 - --out,   Styled audio path

### Dependencies
- python (tested with python 2.7/3.5)
- tensorflow
- librosa

###Demo 

00:00-00:10 - Content - Alan Walker fade

00:10-00:20 - Style- Chain smokers -Don't Let Me Down

00:10-00:30 - Styled song

[Song style transfer AI test](https://www.youtube.com/watch?v=iUujo7i6P3w)

![Spectrum](https://raw.githubusercontent.com/rupeshs/neuralsongstyle/master/plots/spectrum.jpg "Spectrum")

### References
[Audio texture synthesis and style transfe](http://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)

