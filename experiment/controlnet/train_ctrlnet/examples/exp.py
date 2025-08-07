# conda activate controlnet
# python exp.py

import os
import os.path as osp
import platform


# helpers
def add_pythonpath(p):

    if platform.system() == 'Windows':
        SEP = ';'
    elif platform.system() == 'Linux':
        SEP = ':'
    else:
        raise NotImplementedError
    
    pythonpath = os.environ.get('PYTHONPATH')
    pythonpath = p + SEP + pythonpath if pythonpath else p
    os.environ['PYTHONPATH'] = pythonpath


# configuration
HUGGINGFACE_ROOT    = "/public/mlchen/huggingface/"

SD15_PATH           = osp.join(HUGGINGFACE_ROOT, 'runwayml/stable-diffusion-v1-5')
SD21_PATH           = osp.join(HUGGINGFACE_ROOT, "stabilityai/stable-diffusion-2-1")
SD21_BASE_PATH      = osp.join(HUGGINGFACE_ROOT, "stabilityai/stable-diffusion-2-1-base")


OUTPUT_PATH = 'outputs'
DATASET_PATH = '/public/mlchen/data/fusing/fill50k/'
command = f'\
    accelerate launch --main_process_port=29001 train_controlnet.py \
    --pretrained_model_name_or_path={SD15_PATH} \
    --output_dir={OUTPUT_PATH} \
    --train_data_dir={DATASET_PATH} \
    --cache_dir=.cache \
    --learning_rate=1e-5 \
    --validation_image \"./conditioning_image_1.png\" \"./conditioning_image_2.png\" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=8 \
    --max_train_steps=10000 \
    --checkpoints_total_limit=10 \
    --resume_from_checkpoint=latest \
    '
print(command)
os.system(command)

# More detials refer to `https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README.md``