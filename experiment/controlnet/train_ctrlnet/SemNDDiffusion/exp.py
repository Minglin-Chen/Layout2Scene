# conda activate controlnet
# python exp.py

import os
import os.path as osp


# helpers
HF_ROOT = os.environ.get('HF_ROOT')
HF_PATH = lambda p: p if HF_ROOT is None else osp.join(HF_ROOT, p)


############################################################################
# ScanNet++
############################################################################
OUTPUT_PATH         = 'outputs_scannetpp/SemNDDiffusion'
DATASET_PATH        = '/public/mlchen/data/SCANNETPP4GEN/'
validation_prompt   = "'a living room'"
validation_image    = f'"{DATASET_PATH}/data/25692/semantic.png"'

command = f'\
    accelerate launch --main_process_port=29002 train_semnddiffusion.py \
    --dataset_type "SCANNETPP" \
    --pretrained_model_name_or_path={HF_PATH("nexuslrf/nd-diffusion")} \
    --output_dir={OUTPUT_PATH} \
    --train_batch_size=16 \
    --max_train_steps=10000 \
    --checkpointing_steps=1000 \
    --learning_rate=5e-6 \
    --train_data_dir={DATASET_PATH} \
    --proportion_empty_prompts=0 \
    --validation_prompt "{validation_prompt}" \
    --validation_image {validation_image} \
    --validation_steps 500 \
    --scale_lr \
    --resume_from_checkpoint=latest \
    '
print(command)
os.system(command)


############################################################################
# Hypersim
############################################################################
# OUTPUT_PATH         = 'outputs_hypersim/SemNDDiffusion'
# DATASET_PATH        = '/public/mlchen/data/SunRGBD/HYPERSIM4GEN/'
# validation_prompt   = "'a bedroom' 'a living room'"
# validation_image    = f'"{DATASET_PATH}/data/00000/semantic.png" "{DATASET_PATH}/data/00000/semantic.png"'

# command = f'\
#     accelerate launch --main_process_port=29002 train_semnddiffusion.py \
#     --dataset_type "HYPERSIM" \
#     --pretrained_model_name_or_path={HF_PATH("nexuslrf/nd-diffusion")} \
#     --depth_type="inverse" \
#     --output_dir={OUTPUT_PATH} \
#     --train_batch_size=16 \
#     --max_train_steps=10000 \
#     --checkpointing_steps=1000 \
#     --learning_rate=5e-6 \
#     --train_data_dir={DATASET_PATH} \
#     --proportion_empty_prompts=0 \
#     --validation_prompt "{validation_prompt}" \
#     --validation_image {validation_image} \
#     --validation_steps 500 \
#     --scale_lr \
#     --resume_from_checkpoint=latest \
#     '
# print(command)
# os.system(command)


############################################################################
# SunRGBD
############################################################################
# OUTPUT_PATH         = 'outputs_sunrgbd/SemNDDiffusion'
# DATASET_PATH        = '/public/mlchen/data/SunRGBD/SUNRGBD4GEN/'
# validation_prompt   = "'a bedroom' 'a living room'"
# validation_image    = f'"{DATASET_PATH}/data/10309/semantic.png" "{DATASET_PATH}/data/10321/semantic.png"'

# command = f'\
#     accelerate launch --main_process_port=29002 train_semnddiffusion.py \
#     --pretrained_model_name_or_path={HF_PATH("nexuslrf/nd-diffusion")} \
#     --depth_type="inverse" \
#     --output_dir={OUTPUT_PATH} \
#     --train_batch_size=16 \
#     --max_train_steps=10000 \
#     --checkpointing_steps=1000 \
#     --learning_rate=5e-6 \
#     --train_data_dir={DATASET_PATH} \
#     --proportion_empty_prompts=0 \
#     --validation_prompt "{validation_prompt}" \
#     --validation_image {validation_image} \
#     --validation_steps 500 \
#     --scale_lr \
#     --resume_from_checkpoint=latest \
#     '
# print(command)
# os.system(command)