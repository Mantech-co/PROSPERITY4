import numpy as np
import matplotlib.pyplot as plt

values = np.arange(670, 925, 5)
n = len(values)
cdf = np.arange(1, n + 1) / n

plt.figure(figsize=(10, 5))
plt.step(values, cdf, where="post")
plt.xlabel("Value")
plt.ylabel("CDF")
plt.title("Uniform Distribution CDF (670–920, step 5)")
plt.grid(True)
plt.tight_layout()
plt.savefig("manual/round_3/uniform_cdf.png", dpi=150)
plt.show()
