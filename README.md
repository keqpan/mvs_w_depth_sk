# mvs_w_depth_sk
Repository with code for DL and FDS course projects at Skoltech

```
./train.sh 4 ./checkpoints_depth  --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3 --epochs 64
```

To train the modification, when passing low quality depth for the warping at first level:
```
./train_w_depth.sh
```

To inference original model:
```
./test.sh ./checkpoints/model_000015.ckpt --outdir ./outputs --interval_scale 1.06
```

To inference model with our modification:
```
./test_w_depth.sh checkpoints_depth/model_000015.ckpt --outdir casmvsnet_w_depth_res  --interval_scale 1.06
```
