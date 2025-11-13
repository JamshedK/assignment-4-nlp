#!/bin/bash
### Some common partitions
##SBATCH --partition=gpu_a100
##SBATCH --partition=sooner_gpu_test_ada
##SBATCH --partition=sooner_gpu_test
#SBATCH --partition=gpu
##SBATCH --partition=debug_gpu # This partition is currently selected.
#
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
##SBATCH --mem=32G
#SBATCH --mem=32G
##SBATCH --time=12:00:00
##SBATCH --time=00:20:00
#SBATCH --time=02:00:00
#SBATCH --job-name=llm_sst_inference
#SBATCH --mail-user=jamshed.k@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs529314/assignment_4/src/
#SBATCH --output=results/llm_sst_%j.out
#SBATCH --error=results/llm_sst_%j.err

#################################################

# Exit on any error
set -e

export MAMBA_ROOT_PREFIX="/scratch/cs529314/micromamba"
MM="/scratch/cs529314/micromamba/bin/micromamba"
eval "$("$MM" shell hook -s bash)"

echo "Activating environment..."
micromamba activate cs5293-4 || { echo "Failed to activate environment"; exit 1; }

echo "Python version:"
python -V

echo "Testing Python imports..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import outlines; print('Outlines imported successfully')"

echo "Starting LLM SST inference..."
python llm-sst-explain.py || { echo "Python script timed out or failed"; exit 1; }

echo "Done."
