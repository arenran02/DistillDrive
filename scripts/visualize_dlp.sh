export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization_dlp/visualize.py \
	projects/configs/distillation/distilldrive_stage0_label.py \
	--result-path work_dirs/distilldrive_stage0_label/results.pkl