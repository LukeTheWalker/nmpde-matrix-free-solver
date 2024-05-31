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
        return match.group(1)
    else:
        return None

# If no filename passed, exit
if len(sys.argv) < 2:
	print("Usage: python plotter.py <plot_type> [basepath]")
	print("<plot_type>\tdimension, scalability")
	print("[basepath]\toptional path to the directory containing the output files (default: current directory)")
	print("\t\tbasepath must have: output_mf, output_mb, output_mg directories containing the .csv files")
	exit()

if len(sys.argv) > 2:
	basepath = path.abspath(sys.argv[2])
else:
	basepath = path.abspath(".")

print("Base directory: ", basepath)

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
	print(df_dim[put].info())

	## Polynomial degree data
	file = glob(path.join(basepath, out + put, "polynomial_degree_*.csv"))[0]
	print("Analyzing", put, "file for polynomial:")
	print("\t", file)
	df_poly[put] = pd.read_csv(file, comment='#', index_col=0)
	print(df_poly[put].info())



'''
# Set the font size
sns.set_theme(font_scale = 1.3)

# Compose the title from the comments
title = ""
description = "("
with open(filepath, 'r') as f:
	title = f.readline()[1:].strip().replace('_', ' ').capitalize()
	for line in f:
		if line[0] == '#':
			description += line[1:].strip() + ", "
	description = description[:-2] + ")"


sns.set_style("darkgrid")
ax = None

'''

# ====== Plot for dimension ======
if sys.argv[1] == "dimension":
	print("Plotting dimension")

# ====== Plot for polynomial ======
elif sys.argv[1] == "polynomial":
	print("Plotting polynomial")


# ====== If the plot type is not recognized ======
else:
	print("Plot type not supported.")
	exit()


# Make directory for the plots if it doesn't exist
if not os.path.exists(os.path.join(".", "plots")):
	os.makedirs(os.path.join(".", "plots"))

# Save the plot in the output directory
plt.savefig(os.path.join(".", "plots", "filename-----" + ".png"))
