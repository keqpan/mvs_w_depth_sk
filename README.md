# How good MVSNets are at Depth Fusion
[[Report](http://vision.stanford.edu/cs598_spring07/report_templates/egpaper.pdf)]
[[Video](https://youtu.be/dQw4w9WgXcQ)]

This repository contains the
code to reproduce the results of Skoltech DL/FDS course project "How good MVSNets are at Depth Fusion".


## FastMVSNet

```
### Installation
 ```bash
pip install -r requirements.txt
```

### Training
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) from [MVSNet](https://github.com/YoYo000/MVSNet) and unzip it to ```data/dtu```.
* Train the network


    ```python fastmvsnet/train.py --cfg configs/dtu.yaml``` - for original FastMVSNet


    ```python fastmvsnet/train1.py --cfg configs/dtu.yaml``` - for FastMVSNet with gt_depth directly added into the input as another dimension (our modification)
  
    You can change the batch size in the configuration file ```configs/dtu.yaml```.

### Validation
* Download the [rectified images](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) from [DTU benchmark](http://roboimagedata.compute.dtu.dk/?page_id=36) and unzip it to ```data/dtu/Eval```. Be sure to set appropriate paths in ```configs/dtu.yaml``` for validation. You can also set the validation frequency in ```fastmvsnet/config.py```:  _C.TRAIN.VAL_PERIOD
    
* Test with the pretrained model

    ```python fastmvsnet/test.py --cfg configs/dtu.yaml TEST.WEIGHT outputs/pretrained.pth```

### Depth Fusion
Run the depth fusion ```tools/depthfusion.py``` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details.

```bash
python tools/depthfusion.py -f dtu -n flow2
```


## CasMVSNet
### Installation
* Follow the instructions in the original repository.
### Data
* Download preprocessed training/validation [DTU](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
  and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip).
* Generate the [Corrupted DTU](#corrupted-dtu) dataset.
### Training
* Train the original model.
```
./train.sh 4 ./checkpoints_depth  --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3 --epochs 64
```
* Train our modification which uses low quality 'sensor' depth to constrain the hypothesis range at the first stage.
```
./train_w_depth.sh
```
### Testing
* Test the original model.
```
./test.sh ./checkpoints/model_000015.ckpt --outdir ./outputs --interval_scale 1.06
```
* Test our modification.
```
./test_w_depth.sh checkpoints_depth/model_000015.ckpt --outdir casmvsnet_w_depth_res  --interval_scale 1.06
```


## Corrupted DTU
* Download [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from CasMVSNet.
* Generate the train data with `make_train` from `corrupted_dtu/corrupted_dtu.py`.
```python
from pathlib import Path
depth_root = Path('.../Depths_raw')
corrupted_depth_root = Path('.../InputDepths')
make_train(depth_root, corrupted_depth_root)
```
* Download the [test split](/CasMVSNet/lists/dtu/test.txt) from the original DTU [Rectified](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip).
* Generate the test data.
```python
from pathlib import Path
rgb_root = Path('.../Rectified')
depth_root = Path('.../Depths_raw')
corrupted_depth_root = Path('.../InputDepths')
make_test(rgb_root, depth_root, corrupted_depth_root)
```
