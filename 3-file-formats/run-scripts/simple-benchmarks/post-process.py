import os
import sys, argparse
import numpy as np

from glob import glob

# Raw data files can be summarized with CLI 'grep time *tiny*.out'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data",
    choices=["seq", "tiny", "large", "full"],
    help="Slurm output file-string",
)

args = parser.parse_args()

data_tag = args.data
project = os.environ.get("PROJECT_ACCOUNT", "project_462000131")
user = os.environ.get("LUMI_USER", os.environ.get("USER", "anisrahm"))

files = glob(f"/scratch/{project}/{user}/slurm/comp-{data_tag}-*.out")
raw_result = {"HDF5": [], "LMDB": [], "SquashFS": []}
for file_name in files:
    with open(file_name, "r") as fd:
        lines = [line.rstrip("\n") for line in fd if "dataloader time:" in line]
        if not lines:
            continue
        line = lines[-1]
        try:
            file_format, time = line.split(" dataloader time: ")
            raw_result[file_format].append(float(time))
        except Exception:
            continue

result = {
    "HDF5": {"average": 0, "std": 0, "N": 0},
    "LMDB": {"average": 0, "std": 0, "N": 0},
    "SquashFS": {"average": 0, "std": 0, "N": 0},
}

for file_format, times in raw_result.items():
    if len(times) > 0:
        result[file_format]["N"] = len(times)
        result[file_format]["average"] = np.average(times)
        result[file_format]["std"] = np.std(times)


for file_format, result in result.items():
    for i, value in result.items():
        print(f"{file_format}, {i} = {round(value, 2)}")
