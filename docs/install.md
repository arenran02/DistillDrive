# Quick Start

### Set up a virtual environment for distilldrive
```bash
conda create -n distilldrive python=3.8 -y
conda activate distilldrive
```

### Install torch related dependencies
```bash
# highher than cu116
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Install dependency packpages
```bash
git clone https://github.com/YuruiAI/DistillDrive
cd DistillDrive
pip3 install --upgrade pip
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
```

