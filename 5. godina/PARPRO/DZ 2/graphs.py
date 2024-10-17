import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dohvaćanje podataka iz datoteke
data = pd.read_csv("results.txt", sep=",", header=None, names=["Description", "Value"])
num_processors = data[data["Description"].str.contains("Number of processors")]["Value"].astype(int).values
durations_7_tasks = data[data["Description"].str.contains("Duration 7 tasks")]["Value"].astype(float).values
durations_49_tasks = data[data["Description"].str.contains("Duration 49 tasks")]["Value"].astype(float).values
durations_343_tasks = data[data["Description"].str.contains("Duration 343 tasks")]["Value"].astype(float).values

# Izračunavanje ubrzanja i učinkovitosti
T1_7_tasks = durations_7_tasks[0]
T1_49_tasks = durations_49_tasks[0]
T1_343_tasks = durations_343_tasks[0]

speedups_7_tasks = T1_7_tasks / durations_7_tasks
speedups_49_tasks = T1_49_tasks / durations_49_tasks
speedups_343_tasks = T1_343_tasks / durations_343_tasks

efficiencies_7_tasks = speedups_7_tasks / num_processors
efficiencies_49_tasks = speedups_49_tasks / num_processors
efficiencies_343_tasks = speedups_343_tasks / num_processors

# Crtanje grafova
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Ubrzanje
axes[0].plot(num_processors, speedups_7_tasks, marker='o', label='7 zadataka')
axes[0].plot(num_processors, speedups_49_tasks, marker='o', label='49 zadataka')
axes[0].plot(num_processors, speedups_343_tasks, marker='o', label='343 zadataka')
axes[0].plot(num_processors, num_processors, linestyle='--', label='Idealno')
axes[0].set_xlabel('Broj procesora')
axes[0].set_ylabel('Ubrzanje')
axes[0].set_title('Ubrzanje')
axes[0].legend()
axes[0].grid(True)

# Učinkovitost
axes[1].plot(num_processors, efficiencies_7_tasks, marker='o', label='7 zadataka')
axes[1].plot(num_processors, efficiencies_49_tasks, marker='o', label='49 zadataka')
axes[1].plot(num_processors, efficiencies_343_tasks, marker='o', label='343 zadataka')
axes[1].set_xlabel('Broj procesora')
axes[1].set_ylabel('Učinkovitost')
axes[1].set_title('Učinkovitost')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('speedup_efficiency.png')
plt.show()

# Spremanje podataka u CSV datoteku
results = {
    'Broj procesora': num_processors,
    'Trajanje 7 zadataka (s)': durations_7_tasks,
    'Trajanje 49 zadataka (s)': durations_49_tasks,
    'Trajanje 343 zadataka (s)': durations_343_tasks,
    'Ubrzanje 7 zadataka': speedups_7_tasks,
    'Ubrzanje 49 zadataka': speedups_49_tasks,
    'Ubrzanje 343 zadataka': speedups_343_tasks,
    'Učinkovitost 7 zadataka': efficiencies_7_tasks,
    'Učinkovitost 49 zadataka': efficiencies_49_tasks,
    'Učinkovitost 343 zadataka': efficiencies_343_tasks
}

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
