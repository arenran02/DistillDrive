### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir checkpoint
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O checkpoint/resnet50-19c8e357.pth
```

### Commence training and testing
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
The code will be released soon.
```

#### Motion-Guided Student Model
```bash
The code will be released soon.
```

### Visualization
```bash
The code will be released soon.
```