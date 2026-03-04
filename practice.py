import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Task 1: DFT and IDFT
# ==============================

def dft(x):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X


def idft(X):
    X = np.array(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N

    return x


# ==============================
# Signal Definitions (N = 64)
# ==============================

N = 64
n = np.arange(N)

# Rectangular pulse
xrect = np.zeros(N)
xrect[0:N//8] = 1

# Cosine with m=5
m = 5
xcos = np.cos(2 * np.pi * m * n / N)


# ==============================
# Reconstruction Check Function
# ==============================

def reconstruction_test(x, title):
    X = dft(x)
    x_hat = idft(X)

    max_error = np.max(np.abs(x - x_hat))
    l2_error = np.linalg.norm(x - x_hat)
    normalized_error = l2_error / (np.linalg.norm(x) + 1e-12)

    print(f"\n{title}")
    print("Max error:", max_error)
    print("L2 error:", l2_error)
    print("Normalized error:", normalized_error)

    # Plot signal + reconstruction
    plt.figure()
    plt.stem(n, x, basefmt=" ", linefmt="b-", markerfmt="bo", label="x[n]")
    plt.stem(n, np.real(x_hat), basefmt=" ", linefmt="r--", markerfmt="rx", label="Re{x_hat[n]}")
    plt.legend()
    plt.title(f"{title}: Signal vs Reconstruction")

    # Magnitude spectrum
    plt.figure()
    plt.stem(np.abs(X))
    plt.title(f"{title}: |X[k]|")

    # Phase spectrum
    plt.figure()
    plt.stem(np.angle(X))
    plt.title(f"{title}: Phase of X[k]")

    plt.show()


reconstruction_test(xrect, "Rectangular Pulse")
reconstruction_test(xcos, "Cosine Signal")


# ==============================
# Task 2: Circular Convolution
# ==============================

def circular_convolution(x, h):
    x = np.array(x, dtype=complex)
    h = np.array(h, dtype=complex)
    N = len(x)
    y = np.zeros(N, dtype=complex)

    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]

    return y


# Small example (N=4)
x_small = np.array([1, 2, 3, 4])
h_small = np.array([4, 3, 2, 1])

y_time = circular_convolution(x_small, h_small)

X = dft(x_small)
H = dft(h_small)
Y = X * H
y_freq = idft(Y)

print("\nCircular Convolution Theorem Check (N=4)")
print("Max error:", np.max(np.abs(y_time - np.real(y_freq))))

plt.figure()
plt.stem(y_time, linefmt="b-", markerfmt="bo", basefmt=" ", label="Time domain")
plt.stem(np.real(y_freq), linefmt="r--", markerfmt="rx", basefmt=" ", label="Freq domain")
plt.legend()
plt.title("Circular Convolution Comparison")
plt.show()


# ==============================
# Task 3: Cross-Correlation via DFT
# ==============================

def cross_correlation_dft(x, y):
    X = dft(x)
    Y = dft(y)
    R = X * np.conjugate(Y)
    r = idft(R)
    return r


ns = 12
y_shifted = np.roll(xcos, ns)

rxy = cross_correlation_dft(xcos, y_shifted)
rxy_real = np.real(rxy)

n_star = np.argmax(rxy_real)

print("\nCross-Correlation")
print("Estimated shift n*:", n_star)

plt.figure()
plt.stem(rxy_real)
plt.title("Circular Cross-Correlation rxy[n]")
plt.show()