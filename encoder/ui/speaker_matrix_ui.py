from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
from PyQt4 import QtGui
import numpy as np
import librosa
import sys
from .. import audio
from ..params_data import sampling_rate, mel_window_step
from ..preprocess import preprocess_wave


class SpeakerMatrixUI(QtGui.QDialog):
    def __init__(self, speakers, partial_utterances):
        self.speakers = speakers
        self.partial_utterances = partial_utterances
        
        # Create the ui
        app = QtGui.QApplication(sys.argv)
        super().__init__(None)
        
        # Display it and stay in mainloop until the window is closed
        total_utterances = sum(map(len, self.partial_utterances.values()))
        print('Drawing plots for %d utterances, please wait...' % total_utterances)
        self.setup_ui()
        self.show()
        app.exec_()
    
    def setup_ui(self):
        grid = QtGui.QGridLayout()
        
        # Draw the grid
        for i, speaker in enumerate(self.speakers):
            # Speaker ID
            label = QLabel(speaker.name)
            grid.addWidget(label, i, 0)
            
            # Utterances
            for j, partial_utterance in enumerate(self.partial_utterances[speaker]):
                grid.addLayout(self._plot_partial_utterance(partial_utterance), i, j + 1)
        
        self.setLayout(grid)
        
    def _plot_partial_utterance(self, partial_utterance):
        figure = Figure()
        canvas = FigureCanvas(figure)
        
        # Load the partial utterance's frames and waveform
        utterance, frames, frames_range = partial_utterance
        wave_fpath = utterance.wave_fpath
        wave = audio.load(wave_fpath)
        wave = preprocess_wave(wave)
        
        wave_range = np.array(frames_range) * sampling_rate * (mel_window_step / 1000)
        wave = wave[int(wave_range[0]):int(wave_range[1])]
    
        # Plot the spectrogram and the waveform
        ax = figure.add_subplot(211)
        librosa.display.specshow(
            librosa.power_to_db(frames.transpose(), ref=np.max),
            hop_length=int(sampling_rate * 0.01),
            y_axis='mel',
            x_axis='time',
            sr=sampling_rate,
            ax=ax
        )
        ax.get_xaxis().set_visible(False)
        ax = figure.add_subplot(212, sharex=ax)
        librosa.display.waveplot(
            wave,
            sr=sampling_rate,
            ax=ax
        )
        figure.tight_layout()
        canvas.draw()
        
        button = QtGui.QPushButton('Play')
        button.clicked.connect(lambda: audio.play_wave(wave))
    
        layout = QtGui.QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(button)
        return layout


import audio
import numpy as np
from vlibs import fileio
from ui.speaker_matrix_ui import SpeakerMatrixUI
from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader
from data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from config import *

if __name__ == '__main__':
    dataset = SpeakerVerificationDataset(['voxceleb2'])
    loader = SpeakerVerificationDataLoader(dataset, 4, 5, num_workers=3)
    for batch in loader:
        SpeakerMatrixUI(batch.speakers, batch.partial_utterances)
