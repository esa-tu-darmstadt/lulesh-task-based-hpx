import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set plot style
sns.set(style="darkgrid", context="paper", font_scale=1)

# read CSV files
if (len(sys.argv) < 3):
    # assume default location set by execution script
    hpx_result_file = "../results/hpx_results.csv"
    reference_result_file = "../results/omp_results.csv"
else:
    hpx_result_file = sys.argv[1]
    reference_result_file = sys.argv[2]
hpx_df = pd.read_csv(hpx_result_file)
ref_df = pd.read_csv(reference_result_file)

# generate line plots
# filter data frames
hpx_df_filtered = hpx_df[hpx_df['regions'] == 11]
ref_df_filtered = ref_df[ref_df['regions'] == 11]

# group data frames
hpx_df_grouped = hpx_df_filtered.groupby(['size', 'threads'])['runtime'].mean().reset_index()
ref_df_grouped = ref_df_filtered.groupby(['size', 'threads'])['runtime'].mean().reset_index()

# merge data frames
merged_df = pd.merge(hpx_df_grouped, ref_df_grouped, on=['size', 'threads'], suffixes=('_hpx', '_ref'))
plot_df = merged_df[['size', 'threads', 'runtime_hpx', 'runtime_ref']].rename(columns={"threads": "Execution threads", "runtime_hpx": "HPX", "runtime_ref": "OpenMP"})

# prepare for plotting
runtime_min = min(plot_df['HPX'].min(), plot_df['OpenMP'].min())
runtime_max = min(plot_df['HPX'].max(), plot_df['OpenMP'].max())
sizes = plot_df['size'].unique()

# generate plots
for size in sizes:
    plot_df_filtered = plot_df[plot_df['size'] == size]
    melted_df = plot_df_filtered.melt(id_vars=['size', 'Execution threads'], var_name='Implementation', value_name='Runtime (s)')
    plot = sns.lineplot(x="Execution threads", y="Runtime (s)", hue="Implementation", data=melted_df, marker="X")
    plot.set(yscale="log", ylim=(runtime_min / 2, runtime_max * 2), title=f"Problem size = {size}")
    plt.savefig(f"runtimes_{size}.png")
    plt.clf()

# generate bar plot
# filter data frames
hpx_df_filtered = hpx_df[hpx_df['threads'] == 24]
ref_df_filtered = ref_df[ref_df['threads'] == 24]

# group data frames
hpx_df_grouped = hpx_df_filtered.groupby(['size', 'regions'])['runtime'].mean().reset_index()
ref_df_grouped = ref_df_filtered.groupby(['size', 'regions'])['runtime'].mean().reset_index()

# merge data frames
merged_df = pd.merge(hpx_df_grouped, ref_df_grouped, on=['size', 'regions'], suffixes=('_hpx', '_ref'))

# calculate speedup
merged_df['Speedup'] = merged_df['runtime_ref'] / merged_df['runtime_hpx']
plot_df = merged_df[['size', 'regions', 'Speedup']]
merged_df.rename(columns={"size": "Problem size", "regions": "Regions", "runtime_hpx": "Runtime (HPX)", "runtime_ref": "Runtime (OpenMP)"}, inplace=True)
plot = sns.catplot(x="Problem size", y="Speedup", hue="Regions", data=merged_df, kind="bar")
plot.figure.savefig("speedups.png")

print("Speedups of HPX implementation compared to OpenMP reference")
print(merged_df)