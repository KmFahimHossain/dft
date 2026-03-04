import numpy as np

class DiscreteSignal:
    """
    Represents a discrete-time signal.
    """
    def __init__(self, data):
        # Ensure data is a numpy array, potentially complex
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)
               
    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal object.
        """
        current_length = len(self.data)

        if new_length == current_length:
            return DiscreteSignal(self.data.copy())
        elif new_length > current_length:
            padded = np.zeros(new_length, dtype=np.complex128)
            padded[:current_length] = self.data
            return DiscreteSignal(padded)
        else:
            truncated = self.data[:new_length]
            return DiscreteSignal(truncated)

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Required for Task 4 (Drawing App).
        """
        old_length = len(self.data)
        old_indices = np.linspace(0, old_length - 1, old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        real_part = np.interp(new_indices, old_indices, self.data.real)
        imag_part = np.interp(new_indices, old_indices, self.data.imag)
        return DiscreteSignal(real_part + 1j * imag_part)


class DFTAnalyzer:
    """
    Performs Discrete Fourier Transform using O(N^2) method.
    """
    def compute_dft(self, signal: DiscreteSignal):
        """
        Compute DFT using naive summation.
        Returns: numpy array of complex frequency coefficients.
        """
        x = signal.data
        N = len(x)

        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            summation = 0
            for n in range(N):
                angle = -2j * np.pi * k * n / N
                summation += x[n] * np.exp(angle)
            X[k] = summation
        return X

    def compute_idft(self, spectrum):
        """
        Compute Inverse DFT using naive summation.
        Returns: numpy array (time-domain samples).
        """
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)

        x = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            summation = 0
            for k in range(N):
                angle = 2j * np.pi * k * n / N
                summation += X[k] * np.exp(angle)
            x[n] = summation / N
        return x

class FastFourierTransform(DFTAnalyzer):
    def compute_dft(self, signal: DiscreteSignal):
        x = signal.data
        N = len(x)
        if N <= 1:
            return x
        if N & (N - 1) != 0:
            raise ValueError("Input length must be a power of 2 for Radix-2 FFT.")
        return self._fft_recursive(x)

    def _fft_recursive(self, x):
        N = len(x)
        if N == 1:
            return x
        
        even = self._fft_recursive(x[0::2])
        odd = self._fft_recursive(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        first_half = even + factor[:N // 2] * odd
        second_half = even - factor[:N // 2] * odd
        return np.concatenate([first_half, second_half])

    def compute_idft(self, spectrum):
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        conjugated = np.conjugate(X)
        signal = self._fft_recursive(conjugated)
        signal = np.conjugate(signal) / N
        return signal
    