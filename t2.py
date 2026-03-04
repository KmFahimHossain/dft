import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Direct DFT (O(N^2))
# =====================================

def dft(x):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X


# =====================================
# Iterative Radix-2 FFT
# =====================================

def bit_reverse_indices(N):
    bits = int(np.log2(N))
    reversed_indices = np.zeros(N, dtype=int)

    for i in range(N):
        b = format(i, f'0{bits}b')
        reversed_indices[i] = int(b[::-1], 2)

    return reversed_indices


def fft_iterative(x):
    x = np.array(x, dtype=complex)
    N = len(x)

    if N & (N - 1) != 0:
        raise ValueError("Length must be power of 2")

    # Bit-reversal permutation
    indices = bit_reverse_indices(N)
    X = x[indices].copy()

    # Iterative butterfly stages
    size = 2
    while size <= N:
        half = size // 2
        twiddle_base = np.exp(-2j * np.pi / size)

        for start in range(0, N, size):
            for k in range(half):
                twiddle = twiddle_base ** k
                even = X[start + k]
                odd = X[start + k + half] * twiddle

                X[start + k] = even + odd
                X[start + k + half] = even - odd

        size *= 2

    return X


# =====================================
# Test Input
# =====================================

N = 64
n = np.arange(N)

# Example input: mixture of tones
x = np.cos(2 * np.pi * 5 * n / N) + 0.5 * np.sin(2 * np.pi * 12 * n / N)

# =====================================
# Compute Transforms
# =====================================

X_dft = dft(x)
X_fft = fft_iterative(x)

# =====================================
# Compare Results
# =====================================

error = np.max(np.abs(X_dft - X_fft))

print("Max difference between DFT and FFT:", error)

# Plot magnitude comparison
plt.figure()
plt.stem(np.abs(X_dft), basefmt=" ", label="DFT")
plt.stem(np.abs(X_fft), basefmt=" ", linefmt="r--", markerfmt="rx", label="FFT")
plt.legend()
plt.title("DFT vs Iterative FFT (Magnitude)")
plt.show()