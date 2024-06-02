# Script that plots the data from the csv files and saves them as png files
import re
import sys
import os
from os import path
from glob import glob

from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
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
	df_poly[put] = pd.read_csv(file, comment='#')
	print(df_poly[put].info())




# ====== Plot for strong scaling for all the solvers ======
#	Plot the solve time as a function of the number of processes for the largest two values of n_dofs
#	for each of the solvers, with the ideal scaling reference.
#	Then do the same for the setup+assemble time.
if "strongcomp" in sys.argv[1]:
	print("Plotting strong scaling for all solvers")
	# Solve time as a function of the number of processes all in one plot for a single n_dof
	for time_type in ("solve", "setup+assemble"):
		fig, ax = plt.subplots()
		for put in put_types:
			df = df_dim[put]
			# Sort the data by the number of processes
			df = df.sort_values(by='proc')
			# Get the largest two values of n_dofs and plot them
			for dof_value in df['n_dofs'].drop_duplicates().nlargest(2).tolist():
				df1 = df[df['n_dofs'] == dof_value]
				# Compose the label
				lab = put + " (" + str(round(dof_value/1e6, 1)) + "M DoFs)"
				ax.plot(df1['proc'], df1[time_type], label=lab, marker='o', linestyle='-.')

		#Plot ideal scaling
		proc = df['proc']
		solve = df[time_type]
		ax.plot(proc, 1e2 / proc, label="Ideal scaling", linestyle='--', color='black')
		ax.set_xlabel("Number of processors")
		ax.set_ylabel(time_type + " time (s)")
		ax.set_yscale('log')
		ax.grid(True, which="both", ls="--")
		ax.set_xticks(df['proc'].unique())
		ax.set_title("Strong scaling for " + time_type + " time for all the solvers")
		ax.legend()
		plt.savefig(os.path.join(".", "plots", "strongcomp_" + time_type + ".png"))
		print("Strong scaling plot saved in", os.path.join(".", "plots", "strongcomp_" + time_type + ".png"))


# ====== Plot for strong scaling for each solver ======
#	Plot the previous quantities but for all the problem sizes and separately for each
# 	of the solvers.
if "strongsingle" in sys.argv[1]:
	print("Plotting strong scaling for each solver")
	# Solve time as a function of the number of processes all in one plot for a single n_dof
	for time_type in ("solve", "setup+assemble"):
		for put in put_types:
			fig, ax = plt.subplots()
			df = df_dim[put]
			# Sort the data by the number of processes
			df = df.sort_values(by='proc')
			# Get the largest two values of n_dofs and plot them
			for dof_value in df['n_dofs'].drop_duplicates().tolist():
				df1 = df[df['n_dofs'] == dof_value]
				# Compose the label
				lab = put + " (" + str(round(dof_value/1e6, 2)) + "M DoFs)"
				ax.plot(df1['proc'], df1[time_type], label=lab, marker='o', linestyle='-.')

			#Plot ideal scaling
			proc = df['proc']
			solve = df[time_type]
			ax.plot(proc, 1e2 / proc, label="Ideal scaling", linestyle='--', color='black')
			ax.plot(proc, 1e1 / proc, linestyle='--', color='black')
			ax.set_xlabel("Number of processors")
			ax.set_ylabel(time_type + " time (s)")
			ax.set_yscale('log')
			ax.grid(True, which="both", ls="--")
			ax.set_xticks(df['proc'].unique())
			ax.set_title("Strong scaling of " + time_type + " time for " + put)
			ax.legend(loc='upper right')
			plt.savefig(os.path.join(".", "plots", "strong_" + put + "_" + time_type + ".png"))
			print("Strong scaling plot saved in", os.path.join(".", "plots", "strong_" + put + "_" + time_type + ".png"))


# ====== Plot for polynomial ======
#	Plot the MDofs/s metric as a function of the polynomial degree for all the solvers
if "polynomial" in sys.argv[1]:
	print("Plotting polynomial study for all solvers")
	for time_type in ("solve", "setup+assemble"):
		fig, ax = plt.subplots()
		for put in put_types:
			df = df_poly[put]
			ax.plot(df['degree'], 1e-6 * df['dofs'] / df[time_type], label=put, marker='o', linestyle='-.')
		ax.set_ylabel(time_type + " MDofs/s")
		ax.set_xlabel("Polynomial degree")
		ax.set_xticks(df['degree'].unique())
		ax.grid(True, which="both", ls="--")
		ax.set_title("Polynomial degree performance for " + time_type)
		ax.legend()
		plt.savefig(os.path.join(".", "plots", "polynomial_" + time_type + ".png"))
		print("Polynomial plot saved in", os.path.join(".", "plots", "polynomial_" + time_type + ".png"))


# ====== Plot for solve time ======
#	Plot the solve time and setup time as a function of the number of the dofs number for all the solvers
if "time" in sys.argv[1]:
	print("Plotting time study for all solvers")
	for time_type in ("solve", "setup+assemble"):
		fig, ax = plt.subplots()
		for put in put_types:
			df = df_dim[put]
			# Sort the data by the number of processes
			df = df.sort_values(by='n_dofs')
			# Get the largest two values of n_dofs and plot them
			for proc_value in (8, 16, 32):
				df1 = df[df['proc'] == proc_value]
				# Compose the label
				lab = put + " (" + str(proc_value) + " procs)"
				ax.plot(df1['n_dofs'], df1[time_type], label=lab, marker='o', linestyle='-.')

		ax.set_xlabel("Number of DoFs")
		ax.set_ylabel(time_type + " time (s)")
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.grid(True, which="both", ls="--")
		ax.set_title(time_type + " time for all solvers")
		ax.legend()
		plt.savefig(os.path.join(".", "plots", "time_" + time_type + ".png"))
		print("Time plot saved in", os.path.join(".", "plots", "time_" + time_type + ".png"))


if "speedup" in sys.argv[1]:
	print("Plotting solve speedup for mf with respect to mg in function of dofs")
	fig, ax = plt.subplots()
	df_mf = df_dim["mf"]
	df_mg = df_dim["mg"]
	df_mf = df_mf.sort_values(by='n_dofs')
	df_mg = df_mg.sort_values(by='n_dofs')
	for proc_value in (8, 16, 32):
		df1_mf = df_mf[df_mf['proc'] == proc_value]
		df1_mg = df_mg[df_mg['proc'] == proc_value]
		speedup = df1_mg['solve'] / df1_mf['solve']
		ax.plot(df1_mf['n_dofs'], speedup, label=str(proc_value) + " procs", marker='o', linestyle='-.')

	ax.plot(df1_mf['n_dofs'], np.log(df1_mf['n_dofs']) - 8, label="Log speedup", linestyle='--', color='black')
	ax.set_xlabel("Number of DoFs")
	ax.set_ylabel("Speedup")
	ax.set_xscale('log')
	ax.grid(True, which="both", ls="--")
	ax.set_title("Solve speedup for mf with respect to mg")
	ax.legend()
	plt.savefig(os.path.join(".", "plots", "speedup_mf_mg.png"))
	print("Speedup plot saved in", os.path.join(".", "plots", "speedup_mf_mg.png"))



# ====== If the plot type is not recognized ======
if all(s not in sys.argv[1] for s in ("strongcomp", "strongsingle", "polynomial", "speedup", "time")):
	print("Plot type not supported!!!!!")
	print("Run without arguments to see the usage")
	exit()