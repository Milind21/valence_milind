import matplotlib.pyplot as plt
import numpy as np

# X labels
x_labels_100 = [
    '100 P1 0s 3.5', '100 P2 0s SI 3.5', '100 P3 0s SI 3.5', '100 P3 0s SI 4-o-m', '100 P1 0s SI 4-o-m',
    '100 P2 0s SI 4-o-m', '100 P1 1s 3.5', '100 P2 1s 3.5', '100 P3 1s 3.5', '100 P3 1s 4-o-m', '100 P2 1s 4-o-m',
    '100 P1 1s 4-o-m', '100 P1 3s 3.5', '100 P2 3s 3.5', '100 P3 3s 3.5', '100 P3 3s 4-o-m', '100 P2 3s 4-o-m',
    '100 P1 3s 4-o-m', '100 P1 2s 3.5', '100 P2 2s 3.5', '100 P3 2s 3.5', '100 P3 2s 4-o-m', '100 P2 2s 4-o-m',
    '100 P1 2s 4-o-m'
]

x_labels_500 = [
    '500 P3 2s 4-o-m unbiased', '500 P3 2s 3.5 unbiased', '500 P1 2s 4-o-m unbiased', '500 P1 2s 3.5 unbiased',
    '500 P1 3s 3.5 unbiased', '500 P1 3s 4-o-m unbiased', '500 P1 5s 3.5 unbiased', '500 P1 5s 4-o-m unbiased',
    '500 P1 3s 3.5 unbiased RAG', '500 P1 3s 4-o-m unbiased RAG', '500 unbiased RAG', '500 unbiased RAG 2 extract',
    '500 unbiased RAG 3 extract', '500 unbiased RAG 4 extract', '500 unbiased RAG 5 extract'
]

# Values for UAR, Kappa, and Spearman for 100
uar_values_100 = [
    0.2252, 0.1173, 0.1672, 0.1961, 0.1915, 0.0628, 0.1070, 0.2736, 0.1783, 0.1915, 0.1845, 0.2078, 0.2139,
    0.1495, 0.1541, 0.1822, 0.2073, 0.2545, 0.1061, 0.1560, 0.1015, 0.0875, 0.1845, 0.1868
]

kappa_values_100 = [
    0.0034, 0.0495, 0.0222, 0.0195, 0.0018, 0.0087, 0.0286, 0.0286, 0.0103, 0.0257, 0.0097, 0.0376, 0.0158,
    0.0444, 0.0155, 0.0140, 0.0204, 0.0615, 0.0167, 0.0046, 0.1279, 0.0276, 0.0104, 0.0271
]

spearman_values_100 = [
    0.3294, 0.2634, 0.1981, 0.3490, 0.4155, 0.4363, 0.4050, 0.4662, 0.2424, 0.4281, 0.4713, 0.3511, 0.3372,
    0.1031, 0.1604, 0.0887, 0.3221, 0.2254, 0.5522, 0.0442, 0.0718, 0.3787, 0.3038, 0.3196
]

# Values for UAR, Kappa, and Spearman for 500
uar_values_500 = [
    0.1138, 0.0970, 0.0972, 0.0521, 0.0697, 0.0838, 0.0421, 0.0356, 0.1253, 0.1240, 0.5990, 0.5957, 0.2736,
    0.2881, 0.2246
]

kappa_values_500 = [
    0.0140, 0.0017, 0.0124, 0.0154, 0.0031, 0.0181, 0.0133, 0.0335, 0.0290, 0.0143, 0.9869, 0.9867, 0.4670,
    0.5503, 0.1801
]

spearman_values_500 = [
    0.2205, 0.3422, 0.2083, 0.4429, 0.5287, 0.2068, 0.6055, 0.5556, 0.2437, 0.2392, 0.9991, 0.9991, 0.9395,
    0.9462, 0.8800
]

# Create first plot for 100 prompts
plt.figure(figsize=(15, 8))
plt.plot(x_labels_100, uar_values_100, label='UAR', marker='o', color='blue')
plt.plot(x_labels_100, kappa_values_100, label='Kappa', marker='s', color='green')
plt.plot(x_labels_100, spearman_values_100, label='Spearman', marker='^', color='red')
plt.xlabel('Prompts (100)')
plt.ylabel('Metrics')
plt.title('UAR, Kappa, and Spearman Values for 100')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Create second plot for 500 prompts
plt.figure(figsize=(15, 8))
plt.plot(x_labels_500, uar_values_500, label='UAR', marker='o', color='blue')
plt.plot(x_labels_500, kappa_values_500, label='Kappa', marker='s', color='green')
plt.plot(x_labels_500, spearman_values_500, label='Spearman', marker='^', color='red')
plt.xlabel('Prompts (500)')
plt.ylabel('Metrics')
plt.title('UAR, Kappa, and Spearman Values for 500')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
