#!/bin/bash
#SBATCH -p tut 
#SBATCH -n 1
#SBATCH -t 30
#SBATCH --gres=gpu:1
#SBATCH --mem=4096
ef --solver amgx --backend cupy contour.conf