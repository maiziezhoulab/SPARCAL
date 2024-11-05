import pandas as pd
df = pd.read_csv('/data/maiziezhou_lab/hanliu/projects/snv_call/data/reference/tissue_positions_list_GTs.txt', sep='\t')

print(set(df['ground_truth'].tolist()))
print(len(set(df['ground_truth'].tolist())))