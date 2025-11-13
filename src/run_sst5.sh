#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=sst5_sentiment
#SBATCH --output=sst5_%j.out
#SBATCH --error=sst5_%j.err

# Load modules (adjust for your OSCER setup)
module load Python/3.10.4-GCCcore-11.3.0

# Activate environment
source /home/$USER/venvs/nlp_env/bin/activate  # Adjust path

# Set HF token
export HF_TOKEN="your_token_here"  # Replace with your actual token

# Run the script
cd /home/$USER/assignment_4/src  # Adjust path
python llm-sst.py
