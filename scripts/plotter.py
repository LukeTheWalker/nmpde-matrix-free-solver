# Script that plots the data from the csv files and saves them as png files
from re import L
import sys
import os
from os import path

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If no filename passed, exit
if len(sys.argv) < 2:
	print("Please provide on of the plot types:")
	print("\tdimension, scalability")
	exit()

# Find the path of the base out directories
basepath = path.dirname(__file__)
mf_dir_path = path.abspath(path.join(basepath, "..", "build" , "output_mf"))
mb_dir_path = path.abspath(path.join(basepath, "..", "build" , "output_mb"))
mg_dir_path = path.abspath(path.join(basepath, "..", "build" , "output_mg"))

# Aggregate the data into a single or few datasets

'''
# Read the CSV file using pandas
data = pd.read_csv(filepath, comment='#', index_col=0)

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
if not os.path.exists(os.path.join(os.getcwd(), "..", "plots")):
	os.makedirs(os.path.join(os.getcwd(), "..", "plots"))

# Save the plot in the output directory
plt.savefig(os.path.join(os.getcwd(), "..", "plots", "filename-----" + ".png"))
