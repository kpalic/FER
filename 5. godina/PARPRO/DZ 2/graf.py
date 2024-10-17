import matplotlib.pyplot as plt
import pandas as pd

# Učitavanje rezultata iz datoteke
data = pd.read_csv("results.txt", sep=",", header=None, names=["Description", "Value"])
num_processors = data[data["Description"].str.contains("Number of processors")]["Value"].astype(int).values
durations = data[data["Description"].str.contains("Duration")]["Value"].astype(float).values

# Izračunavanje ubrzanja i učinkovitosti
T1 = durations[0]
speedups = T1 / durations
efficiencies = speedups / num_processors

# Prikaz ubrzanja
plt.figure()
plt.plot(num_processors, speedups, marker='o', label='Measured')
plt.plot(num_processors, num_processors, linestyle='--', label='Ideal')

plt.title('Speedup')
plt.legend()
plt.grid(True)
plt.savefig('speedup.png')

# Prikaz učinkovitosti
plt.figure()
plt.plot(num_processors, efficiencies, marker='o')

plt.title('Efficiency')
plt.grid(True)
plt.savefig('efficiency.png')

plt.show()
