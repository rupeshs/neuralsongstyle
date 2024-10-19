# Neural Song Style
Transferring the style from one song onto another using artificial intelligence.

This is based on the implementation [neural-style-audio](https://github.com/DmitryUlyanov/neural-style-audio-tf) by Dmitry and Vadim.

### Options
 - --content ,Content audio path
 - --style, Style audio path
 - --out,   Styled audio path

### Dependencies
- python (tested with python 2.7/3.5)
- torch
- librosa
- tkinter

### Using the tkinter GUI
A tkinter-based GUI is provided for easier interaction with the neural song style transfer process. The GUI allows users to select content and style audio files, specify the output file path, and start the style transfer process with a single click.

#### Instructions
1. Run the `tkinter_gui.py` script to launch the GUI.
2. Use the "Browse" buttons to select the content and style audio files.
3. Specify the output file path where the styled audio will be saved.
4. Click the "Start Style Transfer" button to begin the process.
5. The progress of the style transfer will be displayed in the GUI.

### Demo 

00:00-00:10 - Content - Alan Walker fade

00:10-00:20 - Style- Chain smokers -Don't Let Me Down

00:10-00:30 - Styled song

[Song style transfer AI test](https://www.youtube.com/watch?v=iUujo7i6P3w)

![Spectrum](https://raw.githubusercontent.com/rupeshs/neuralsongstyle/master/plots/spectrum.jpg "Spectrum")

### References
[Audio texture synthesis and style transfe](http://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)
