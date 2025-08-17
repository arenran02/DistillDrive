export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/stage2/distilldrive_stage2_label.py \
	--result-path work_dirs/distilldrive_stage2_label/results.pkl
