#!/bin/sh
#BSUB -J FT
#BSUB -o FoodTransformer%J.out
#BSUB -e FoodTransformer%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -W 24:00
#BSUB -N
# end of BSUB options

module load python3/3.9.6

# load CUDA (for GPU support)
module load cuda/11.8

# activate the virtual environment
source food/bin/activate

python src/models/train_model.py --image_size 224 --model_name "models/best_model_ever_1.pt" --mode 1 --pretrained
