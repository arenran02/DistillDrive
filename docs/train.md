### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir checkpoint
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O checkpoint/resnet50-19c8e357.pth
```

### Commence training and testing
[Model weights](https://huggingface.co/RuiYuStudying/DistillDrive/tree/main) are published  on Hugging Face.

#### Traing IRL-based Teacher Model
```bash
# train
## Label supervision
sh scripts/train.sh projects/configs/stage0/distilldrive_stage0_label.py 8
## Distribution supervision
sh scripts/train.sh projects/configs/stage0/distilldrive_stage0_distribution.py 8

# test
## Label supervision
sh scripts/test.sh \
    projects/configs/stage0/distilldrive_stage0_label.py \
    checkpoint/distilldrive_stage0_label.pth \
    8
## Distribution supervision
sh scripts/test.sh \
    projects/configs/stage0/distilldrive_stage0_distribution.py \
    checkpoint/distilldrive_stage0_distribution.pth \
    8
```

#### Traing Perception Model
```bash
# train
## Adamax Optimizer
sh scripts/train.sh projects/configs/stage1/distilldrive_stage1_adamax.py 8
## SOAP Optimizer for better performance
sh scripts/train.sh projects/configs/stage1/distilldrive_stage1_soap.py 8

# test
## Adamax Optimizer
sh scripts/test.sh \
    projects/configs/stage1/distilldrive_stage1_adamax.py \
    checkpoint/distilldrive_stage1_adamax.pth \
    8
## SOAP Optimizer for better performance
sh scripts/test.sh \
    projects/configs/stage1/distilldrive_stage1_soap.py \
    checkpoint/distilldrive_stage1_soap.pth \
    8
```

#### Motion-Guided Student Model
```bash
# train
## Label supervision
sh scripts/train.sh projects/configs/stage2/distilldrive_stage2_label.py 8
## Distribution supervision
sh scripts/train.sh projects/configs/stage2/distilldrive_stage2_distribution.py 8

# test
## Adamax Optimizer
sh scripts/test.sh \
    projects/configs/stage2/distilldrive_stage2_label.py \
    checkpoint/distilldrive_stage2_label.pth \
    8
## SOAP Optimizer for better performance
sh scripts/test.sh \
    projects/configs/stage2/distilldrive_stage2_distribution.py \
    checkpoint/distilldrive_stage2_distribution.pth \
    1
```

### Visualization
```bash
# Visualize Teacher DLP
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization_dlp/visualize.py \
	projects/configs/stage0/distilldrive_stage0_label.py \
	--result-path work_dirs/distilldrive_stage0_label/results.pkl

 # Visualize DistillDrive
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/stage2/distilldrive_stage2_label.py \
	--result-path work_dirs/distilldrive_stage2_label/results.pkl
```
