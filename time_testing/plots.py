import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the main data CSV and the mapping CSV
df = pd.read_csv('data_new.csv')
mapping = pd.read_csv('mapping.csv')

# Merge mapping info into the main dataframe
df = df.merge(mapping, on='flag_label')

# Compute mean runtime per program and flag_label
mean_df = (
    df
    .groupby(['program', 'flag_label', 'points', 'VariableNum'])['time_seconds']
    .mean()
    .reset_index()
)
#divide each values in mean_df, for its number of points, last element divide by 32
mean_df['time_seconds'] = mean_df['time_seconds'] / mean_df['points']

mean_df.loc[(mean_df['program'] == 'Auto_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'time_seconds'] *= mean_df.loc[(mean_df['program'] == 'Auto_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'points']
mean_df.loc[(mean_df['program'] == 'Auto_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'time_seconds'] /= 42
mean_df.loc[(mean_df['program'] == 'Bar_NoFlags_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'time_seconds'] *= mean_df.loc[(mean_df['program'] == 'Bar_NoFlags_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'points']
mean_df.loc[(mean_df['program'] == 'Bar_NoFlags_NoMax_NoSym') & (mean_df['flag_label'] == '6U6F'), 'time_seconds'] /= 32



#sort by VariableNum and time_seconds
mean_df = mean_df.sort_values(['VariableNum', 'flag_label', 'time_seconds'])
#print(mean_df[['program', 'flag_label', 'time_seconds']])
flag_order = np.array(['3U3F', '4U4F', '5U5F', '4U4F2CI', '5U5F1CI','6U6F'])
print(flag_order)
# Create a scatter plot: x-axis = flag_label, y-axis = time_seconds
fig, ax = plt.subplots(figsize=(10, 6))
for program, group in mean_df.groupby('program'):
    #want the one in the same program to be connected and based on flag_label
    group = group.sort_values('flag_label', key=lambda x: x.map({label: i for i, label in enumerate(flag_order)}))
    ax.plot(group['flag_label'], group['time_seconds'], marker='o', label=program)  
ax.set_xlabel('Flag Label')
ax.set_ylabel('Average Runtime (seconds)')
ax.set_title('Average Runtime vs Flag Label')
ax.legend()
#put in log scale
ax.set_yscale('log')
# Set x-ticks to be the flag labels in the order defined by flag_order
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xticks(flag_order)
ax.set_yticks([0.1, 1, 10, 100, 1000])
plt.tight_layout()
plt.savefig('average_runtime_vs_flag_label.png')

#add a speed up column to mean_df dividing each time_seconds by the one in program 'Auto_NoMax_NoSym' with the same flag_label
for flag in flag_order:
    #if Auto_NoMax_NoSym and flag_label exists, calculate speedup
    if not ((mean_df['program'] == 'Auto_NoMax_NoSym') & (mean_df['flag_label'] == flag)).any():
        mean_df.loc[mean_df['flag_label'] == flag, 'speedup'] = np.nan
        continue
    auto_time = mean_df.loc[(mean_df['program'] == 'Auto_NoMax_NoSym') & (mean_df['flag_label'] == flag), 'time_seconds'].values[0]
    mean_df.loc[mean_df['flag_label'] == flag, 'speedup'] = auto_time/mean_df.loc[mean_df['flag_label'] == flag, 'time_seconds'] 
print(mean_df[['program', 'flag_label', 'time_seconds', 'speedup']])

#print speedup between diff programs on same flag_label divide all time of a defined flag_label by the one in program 'All'
speedup_df = mean_df.pivot(index='flag_label', columns='program', values='time_seconds')
speedup_df = speedup_df.div(speedup_df['All'], axis=0)
speedup_df = speedup_df.reset_index()
speedup_df.to_csv('speedup.csv', index=False)

#Reorder columsns position based on values in first row
speedup_df = speedup_df[['flag_label'] + sorted(speedup_df.columns[1:], key=lambda x: speedup_df[x].iloc[0])]
print(speedup_df)
# Create a bar plot for speedup
fig, ax = plt.subplots(figsize=(10, 6))
#set the order as the flag_order
speedup_df['flag_label'] = pd.Categorical(speedup_df['flag_label'], categories=flag_order, ordered=True)
speedup_df = speedup_df.sort_values('flag_label', key=lambda x: x.map({label: i for i, label in enumerate(flag_order)}))
speedup_df.set_index('flag_label').plot(kind='bar', ax=ax, logy=True)
ax.set_xlabel('Flag Label')
ax.set_ylabel('Speedup (relative to All)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_title('Speedup of Programs Relative to All')
ax.legend(title='Program')
plt.tight_layout()
plt.savefig('speedup_vs_flag_label.png')