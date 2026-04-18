import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('round 1/leaderboard_data_algo.csv', header=None, names=['position', 'empty', 'id', 'team', 'country', 'score'])
df = df[df['score'] > 0]

plt.figure(figsize=(10, 6))
plt.scatter(df['position'], df['score'], alpha=0.5, s=10)
plt.xscale('log')
plt.title('Score vs. Log(Position)')
plt.xlabel('Position')
plt.ylabel('Score')
plt.grid(True)
plt.show()
