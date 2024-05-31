# Script that plots the data from the csv files and saves them as png files
import re
import sys
import os
from os import path
from glob import glob

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract the number of processes from the filename
def extract_proc(filepath):
	match = re.search(r'_([^_]*)\.csv', filepath)
	if match:
		return int(match.group(1))
	else:
		return None

# If no filename passed, exit
if len(sys.argv) < 2:
	print("Usage: python plotter.py <plot_type> [basepath]")
	print("<plot_type>\tstrong, weak, polynomial")
	print("[basepath]\toptional path to the directory containing the output files (default: current directory)")
	print("\t\tbasepath must have: output_mf, output_mb, output_mg directories containing the .csv files")
	exit()

if len(sys.argv) > 2:
	basepath = path.abspath(sys.argv[2])
else:
	basepath = path.abspath(".")

print("Base directory: ", basepath)

# Make directory for the plots if it doesn't exist
if not os.path.exists(os.path.join(".", "plots")):
	os.makedirs(os.path.join(".", "plots"))

# Name of the output directories as out + put for put in put_types
out = "output_"
put_types = ("mf", "mb", "mg")


# Collection of dfs
df_dim = {}
df_poly = {}

# Load the data from the csv files
for put in put_types:
	## Dimension time data
	file_list = glob(path.join(basepath, out + put, "dimension_time_*.csv"))
	print("Analyzing", put, "files for dimension:")
	for file in file_list: print("\t", file)
	data = []

	# For each dimension time file, read the data and append it to the list
	for file in file_list:
		df_temp = pd.read_csv(file, comment='#')
		df_temp['proc'] = extract_proc(file)
		data.append(df_temp)

	# Concatenate the dataframes
	df_dim[put] = pd.concat(data, ignore_index=True)
	# Correct eventual errors in the column number of setup+assemble time
	if 'steup+assemble' in df_dim[put].columns:
		df_dim[put].rename(columns={'steup+assemble': 'setup+assemble'}, inplace=True)
	
	print(df_dim[put].info())

	## Polynomial degree data
	file = glob(path.join(basepath, out + put, "polynomial_degree_*.csv"))[0]
	print("Analyzing", put, "file for polynomial:")
	print("\t", file)
	df_poly[put] = pd.read_csv(file, comment='#', index_col=0)
	print(df_poly[put].info())




# ====== Plot for strong scaling ======
#	Plot the solve time as a function of the number of processes for the largest two values of n_dofs
#	for each of the solvers, with the ideal scaling reference.
#	Then do the same for the setup+assemble time.
if "strong" in sys.argv[1]:
	print("Plotting strong scaling")
	# Solve time as a function of the number of processes all in one plot for a single n_dof
	for time_type in ("solve", "setup+assemble"):
		fig, ax = plt.subplots()
		for put in put_types:
			df = df_dim[put]
			# Sort the data by the number of processes
			df = df.sort_values(by='proc')
			# Get the largest two values of n_dofs and plot them
			for dof_value in df['n_dofs'].drop_duplicates(inplace=False).nlargest(2).tolist():
				df1 = df[df['n_dofs'] == dof_value]
				# Compose the label
				lab = put + " (" + str(round(dof_value/1e6, 1)) + "M DoFs)"
				ax.plot(df1['proc'], df1[time_type], label=lab, marker='o', linestyle='-.')

		#Plot ideal scaling
		proc = df['proc']
		solve = df[time_type]
		ax.plot(proc, 1e2*solve[0] / proc, label="Ideal scaling", linestyle='--', color='black')
		ax.set_xlabel("Number of processors")
		ax.set_ylabel(time_type + " time (s)")
		ax.set_yscale('log')
		ax.grid(True, which="both", ls="--")
		ax.set_xticks(df['proc'].unique())
		ax.set_title("Strong scaling for " + time_type + " time")
		ax.legend()
		plt.savefig(os.path.join(".", "plots", "strong_" + time_type + ".png"))
		print("Strong scaling plot saved in", os.path.join(".", "plots", "strong_" + time_type + ".png"))


# ====== Plot for polynomial ======
if "polynomial" in sys.argv[1]:
	print("Plotting polynomial")


# ====== If the plot type is not recognized ======
if all(s not in sys.argv[1] for s in ("strong", "polynomial")):
	print("Plot type not supported!!!!!")
	print("Run without arguments to see the usage")
	exit()


# Save the plot in the output directory

