import pandas as pd

COLS = ['position', 'delta_position', 'uid', 'name', 'country_code', 'score']

overall = pd.read_csv('leaderboard_data_overall.csv', header=None, names=COLS).set_index('uid')
manual  = pd.read_csv('leaderboard_data_manual.csv',  header=None, names=COLS).set_index('uid')

common = overall.index.intersection(manual.index)
print(f"overall: {len(overall)}  manual: {len(manual)}  matched: {len(common)}  unmatched: {len(overall)-len(common)}")

algo = overall.loc[common].copy()
algo['score'] = overall.loc[common, 'score'] - manual.loc[common, 'score']

algo = algo.sort_values('score', ascending=False).reset_index()
algo['position'] = range(1, len(algo) + 1)
algo['delta_position'] = ''

algo[COLS].to_csv('leaderboard_data_algo.csv', header=False, index=False)
print(f"written {len(algo)} rows to leaderboard_data_algo.csv")
print(algo[['position', 'name', 'country_code', 'score']].head(10).to_string(index=False))
