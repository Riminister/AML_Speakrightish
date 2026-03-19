import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class Preprocessing:
    def __init__(self):
        self.audio_dir = audio_dir
        self.labels_csv = labels_csv
    

    def zero_crossing_rate(x):
        """
        Compute the zero-crossing rate for a 1D audio signal.
        
        Parameters
        ----------
        x : NumPy Array
            Audio signal. Shape (n_samples,).
            
        Returns
        -------
        float
            Zero-crossing rate (range [0, 1]).
        """
        x = np.asarray(x, dtype=float)

        x_sign = np.where(x>0, 1, 0)
        x_diff = np.diff(x_sign)
        return np.mean(np.abs(x_diff))
    
    """
    DIRECTLY TAKEN FROM AML NOTEBOOK
    """
    def spectral_centroid(x, sr):
        """
        Compute the spectral centroid of an audio signal, in Hz.
        The centroid is computed once for the entire signal.

        Formula:
            centroid = sum(f_k * |X_k|) / sum(|X_k|)
        where X_k is the FFT and f_k are bin center frequencies. |X| denotes the absolute of X

        Args:
            x : NumPy Array
                Audio signal. Shape (n_samples,).
            sr (int or float): Sampling rate in Hz.

        Returns:
            float: Spectral centroid in Hz.

        DIRECTLY TAKEN FROM AML NOTEBOOK
        """
        # Real FFT and frequency bins
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)

        # Take absolute
        X_abs = np.abs(X)

        # Find denominator
        denom = X_abs.sum()

        if denom <= 1e-12: # If input is silent, return 0
            return 0.0 
        
        # Calculate spectral centroid
        centroid_hz = (f * X_abs).sum() / denom
        return centroid_hz
