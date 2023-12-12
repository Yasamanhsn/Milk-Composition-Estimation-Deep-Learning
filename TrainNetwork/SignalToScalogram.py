

def signal_to_scalogram(signal):
    """
    Convert the input signal into a scalogram using Continuous Wavelet Transform (CWT) with the Morlet wavelet.

    Parameters:
    signal (array-like): Input signal as a 1-D array or list.

    Returns:
    array-like: A 2-D array representing the scalogram obtained through Continuous Wavelet Transform.
    """

    import numpy as np
    import pywt
    import matplotlib.pyplot as plt
    import scipy.signal as Signal

    signal = np.array(signal)  # Convert signal to numpy array

    signal_length = len(signal)

    # Set parameters for CWT
    voices_per_octave = 48
    octaves = np.log2(signal_length) / voices_per_octave
    scales = (2 ** (np.arange(0, octaves + 1)))
    scales = np.rint(scales).astype(np.int32)  # Round

    # Calculate filter bank scales
    fb_scales = scales * voices_per_octave + 1

    # Perform Continuous Wavelet Transform (CWT)
    coeffs, _ = pywt.cwt(signal, fb_scales, 'morl')

    # Obtain scalogram by taking absolute values of CWT coefficients
    scalogram = np.abs(coeffs)

    # Convert to log scale (dB) and normalize
    scalogram = 20 * np.log10(scalogram / scalogram.max())
    scalogram = (scalogram + 20) / 40
    #

    # # plot and Save the image to a variable
    # # Create a figure without displaying it
    # plt.figure(dpi=100)
    # plt.imshow(scalogram, aspect='auto', origin='lower')
    # plt.show()


    sampling_rate = 128

    # ------------------  Compute STFT
    # f, t,scalogram = Signal.stft(signal, fs=sampling_rate)
    # extend = [t.min(), t.max(), f.min(), f.max()]
    #
    # import pywt
    #
    # # Define scales for the wavelet transform
    # scales = np.arange(1, 512)
    # # Choose a wavelet (e.g., 'morl' Morlet wavelet)
    # wavelet = 'morl'
    # # Perform Continuous Wavelet Transform (CWT)
    # scalogram, frequencies = pywt.cwt(signal, scales, wavelet)
    #
    # # Get magnitude spectrum
    # scalogram = np.abs(scalogram)
    #
    # # Convert to log scale (dB) and normalize
    # # scalogram = 20 * np.log10(scalogram / scalogram.max())

    #  normalize
    # scalogram = (scalogram + 20) / 40


    # # -----------------------------------
    # # WVD function
    # fs = 64
    #
    # def wv(x, t, f):
    #     tau = np.arange(-len(x) + 1, len(x))
    #     wt = np.zeros((len(t), len(f)))
    #     for i in range(len(t)):
    #         for j in range(len(f)):
    #             for k in tau:
    #                 idx1 = int(i + k / 2)
    #                 idx2 = int(i - k / 2)
    #                 s = x[idx1] * np.conjugate(x[idx2]) * np.exp(-1j * 2 * np.pi * f[j] * k)
    #                 wt[i, j] += s
    #     return wt
    #
    # # Frequency axis
    # fs = 100  # Sampling frequency
    # T = 1  # Time period
    # f0 = 5  # Signal frequency
    #
    # # Time axis
    # t = np.linspace(-T / 2, T / 2, fs * T)
    # freqs = np.linspace(-fs / 2, fs / 2, fs)
    #
    # # Compute WVD
    # scalogram = wv(signal, t, freqs)
    #
    # # Get magnitude spectrum
    # scalogram = np.abs(scalogram)
    #
    # # Convert to log scale (dB) and normalize
    # scalogram = 20 * np.log10(scalogram / scalogram.max())
    #
    #  # normalize
    # scalogram = (scalogram + 20) / 40

    #-----------------------------------
    # -----------------------------------fsst
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from ssqueezepy import ssq_cwt, ssq_stft, stft
    #
    # # scalogram = stft(signal)
    # Twxo, scalogram, *_ = ssq_cwt(signal)
    # # not to use =>   Twxo, scalogram, *_ = ssq_stft(signal)
    # scalogram = np.abs(scalogram)
    # scalogram = 20 * np.log10(scalogram / scalogram.max())
    # scalogram = (scalogram + 20) / 40

    # plt.figure(dpi=100)
    # plt.imshow(scalogram, aspect='auto', origin='lower', cmap='turbo')
    # plt.show()






    # ----------------------------------

    # plot and Save the image to a variable
    # Create a figure without displaying it
    # plt.figure(dpi=100)
    # plt.imshow(scalogram, aspect='auto', origin='lower')
    # plt.show()

    return scalogram
