#!/bin/bash
#SBATCH --mem=10000
#SBATCH --time=230:59:59
#SBATCH -n 1
#SBATCH --mail-type=END

set -e

module purge; module load bluebear # this line is required
module load bear-apps/2019b
module load ngspice/31-foss-2019b
module load Python/3.7.4-GCCcore-8.3.0
module load SciPy-bundle/2019.10-foss-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4

python main.py RC_CRJ 1 Narma CHEN
