import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Direct DFT / IDFT
# ==============================

def dft(x):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X


# ==============================
# Setup
# ==============================

N = 64
n = np.arange(N)

# Two signals for linearity
x1 = np.cos(2 * np.pi * 3 * n / N)
x2 = np.sin(2 * np.pi * 7 * n / N)

a = 2.5
b = -1.2

# ==============================
# 1) Linearity Test
# ==============================

left = dft(a * x1 + b * x2)
right = a * dft(x1) + b * dft(x2)

error_linearity = np.max(np.abs(left - right))

print("Linearity Property")
print("Max error:", error_linearity)

plt.figure()
plt.stem(np.abs(left), basefmt=" ", label="DFT{a x1 + b x2}")
plt.stem(np.abs(right), basefmt=" ", linefmt="r--", markerfmt="rx", label="aX1 + bX2")
plt.legend()
plt.title("Linearity Check (Magnitude)")
plt.show()


# ==============================
# 2) Time Shift Property (Manual Shift)
# ==============================

n0 = 10

x = np.cos(2 * np.pi * 5 * n / N)
X = dft(x)

# Manual circular shift
y = np.zeros(N)

for i in range(N):
    y[i] = x[(i - n0) % N]

Y = dft(y)

# Theoretical frequency-domain result
k = np.arange(N)
phase_factor = np.exp(-2j * np.pi * k * n0 / N)
Y_theory = X * phase_factor

error_shift = np.max(np.abs(Y - Y_theory))

print("\nTime Shift Property")
print("Max error:", error_shift)

# Magnitude comparison
plt.figure()
plt.stem(np.abs(X), basefmt=" ", label="|Original X|")
plt.stem(np.abs(Y), basefmt=" ", linefmt="r--", markerfmt="rx", label="|Shifted Y|")
plt.legend()
plt.title("Magnitude Unchanged After Circular Shift")
plt.show()

# Phase comparison
plt.figure()
plt.stem(np.angle(Y), basefmt=" ", label="Actual Phase")
plt.stem(np.angle(Y_theory), basefmt=" ", linefmt="r--", markerfmt="rx", label="Theoretical Phase")
plt.legend()
plt.title("Phase Rotation Verification")
plt.show()