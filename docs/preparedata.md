### Prepare the data
Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) along with the CAN bus expansion, place the CAN bus expansion in the designated storage directory ${DATA_PATH}, and create symbolic links pointing to `./data/nuscenes
```bash
mkdir data
ln -s ${DATA_PATH} ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```