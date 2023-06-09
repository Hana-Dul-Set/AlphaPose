dcn
non-dcn

python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth --indir ../{INPUT_FILE_NAME}

python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --indir ../{INPUT_FILE_NAME}

--save_img option

python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth --indir ../{INPUT_FILE_NAME} --save_img

python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --indir ../{INPUT_FILE_NAME} --save_img


command 양식
python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth {INPUT_DIR_PATH} --outdir {OUTPUT_DIR_PATH}

## Installation
Please check out [docs/INSTALL.md](docs/INSTALL.md)

## Model Zoo
<details><summary>Model zoo</summary>
    # MODEL ZOO

## MSCOCO dataset 

| Model                    | Backbone | Detector | Input Size |     AP     | Speed |  Download | Config | Training Log |  
|--------------------------|----------|----------|------------|------------|-------|-----------|--------|--------------|
|[Simple Baseline](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | ResNet50           | YOLOv3 | 256x192            | 70.6        | 2.94 iter/s | [model](https://drive.google.com/open?id=1nxyfUbvWDaaT9eDM7Y31ScSVPlGy6gfw) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml)    | [log](logs/simple_res50_256x192.log) |
|[Fast Pose](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | ResNet50           | YOLOv3 | 256x192            | 72.0        | 3.54 iter/s | [model](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml)    | [log](logs/fast_res50_256x192.log) |
|[Fast Pose (DUC)](../configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml)    | ResNet50 - unshuffle           | YOLOv3 | 256x192            | 72.4        | 2.91 iter/s | [model](https://drive.google.com/open?id=1RlnL_YnnmgOM4L9L2szFdUnTjwptqtL-) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml)    | [log](logs/fast_421_res50-shuffle_256x192.log) |
|[HRNet](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | HRNet-W32           | YOLOv3 | 256x192            | 72.5        | 2.13 iter/s | [model](https://drive.google.com/open?id=1i63BPlOnp2vSjIZ7ni4Yp3RCPQwqe922) | [cfg](../configs/coco/hrnet/256x192_w32_lr1e-3.yaml)    | [log](logs/hrnet_w32_256x192.log) |
|[Fast Pose (DCN)](../configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | 72.8        | 2.94 iter/s | [model](https://drive.google.com/open?id=1zUz9YIk6eALCbZrukxD7kQ554nhi1pVv) | [cfg](../configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml)    | [log](logs/fast_dcn_res50_256x192.log) |
|[Fast Pose (DUC)](../configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | 73.3        | 1.62 iter/s | [model](https://drive.google.com/open?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9) | [cfg](../configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | [log](logs/fast_421_res152_256x192.log) |

#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- One TITAN XP is used for speed test, with `batch_size=64` in each iteration.
- Offline human detection results are used in speed test.
- `FastPose` is our own network design. Paper coming soon!


## [Halpe dataset](https://github.com/Fang-Haoshu/Halpe-FullBody) (26 keypoints)

| Model                    | Backbone | Detector | Input Size |     AP     | Speed |  Download | Config |  
|--------------------------|----------|----------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml)    | ResNet50           | YOLOv3 | 256x192            | -        | 13.12 iter/s | [Google](https://drive.google.com/file/d/1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb/view?usp=sharing) [Baidu](https://pan.baidu.com/s/1lvzMhoYgS6o6n8lVDx3GtQ) | [cfg](../configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml)    |

For example, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir examples/demo/ --save_img
```

#### Notes
- This model is trained based on the first 26 keypoints of Halpe Full-body datatset (without face and hand keypoints).
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.


## Multi Domain Models **(Strongly Recommended)** 
| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config | #keypoints |
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|--------------|
|[Fast Pose](../configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 50.1       | 16.28 iter/s | [Google](https://drive.google.com/file/d/1Bb3kPoFFt-M0Y3ceqNO8DTXi1iNDd4gI/view?usp=sharing) [Baidu(code: d0wi)](https://pan.baidu.com/s/1GaHzMHTqYze2rVn7u1sjVg) | [cfg](../configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | 136 |
|[Fast Pose (DCN)](../configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined (10 hand weight) | 49.8        | 10.35 iter/s | [Google](https://drive.google.com/file/d/1wX1Z2ZOoysgSNovlgiEtJKpbR8tUBWYR/view?usp=sharing) [Baidu(code: app1)](https://pan.baidu.com/s/1bIro0XfYj0FIVf84QzdDoQ) | [cfg](../configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | 136 |
|[Fast Pose (DCN)](../configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | -        | 13.88 iter/s | [Google](https://drive.google.com/file/d/14Qn9gxm-EVzqFi7v25Y5TqKIvrFLy_BR/view?usp=sharing) [Baidu(code: 6kwr)](https://pan.baidu.com/s/1GLNxN3gfekUVY0HZu41fJQ) | [cfg](../configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | 68 (no face) |
|[Fast Pose (DCN)](../configs/single_hand/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | - | 256x192            | Symmetric Integral | -        | 30.20 iter/s | [Google](https://drive.google.com/file/d/1MntndimlUP5Hxef1UN9ZDMBVglfA606J/view?usp=sharing) [Baidu(code: nwxx)](https://pan.baidu.com/s/1OR-uH25MFQ7kY8Gt_aJfbw ) | [cfg](../configs/single_hand/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | 21 (single hand) |

For the most accurate wholebody pose estimation, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml --checkpoint pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth --indir examples/demo/ --save_img
```
or, you can run with (this version is a little faster and more accurate on body keypoints, but its performance on hand keypoints is worser):
```
python scripts/demo_inference.py --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --indir examples/demo/ --save_img
```
#### Notes
- Above models are trained on multiple datasets, thus they can perform well for in-the-wild images.
- 'Combined (10 hand weight)' means that we use different loss for hand and body keypoints

## [Halpe dataset](https://github.com/Fang-Haoshu/Halpe-FullBody) (136 keypoints)

| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config |
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x.yaml)    | ResNet50           | YOLOv3 | 256x192            | Heatmap | 41.7      | 4.37 iter/s | [Google](https://drive.google.com/file/d/1LbKM2TOxKdpIZoDxCo6ldmOf62pw6z8A/view?usp=sharing) [Baidu(code: y8a0)](https://pan.baidu.com/s/1z1xKIyyet5y-rr7ZQSNX_A) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 44.1  | 16.50 iter/s | [Google](https://drive.google.com/file/d/1_10JYI3O-VbrAiONfL36UxLf9UXMoUYA/view?usp=sharing) [Baidu(code: 9e4z)](https://pan.baidu.com/s/1lakMQbqIWdNV_Khm8Hfcpw) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Symmetric Integral | 46.2      | 16.58 iter/s | [Google](https://drive.google.com/file/d/1S49aDYGVjEJpx4MnFu7TFzzsbp7Si6h_/view?usp=sharing) [Baidu(code: 0yyf)](https://pan.baidu.com/s/1Xx2XJLrds80tp9QEQclR_A) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | 45.4        | 10.07 iter/s | [Google](https://drive.google.com/file/d/1jt-V1Zh-eYgX_-2mrBTV9Ip6z7JjApEC/view?usp=sharing) [Baidu(code: hln3)](https://pan.baidu.com/s/1yZNora5LhH-6eeTEw2S15w) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DCN)](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined (10 hand weight) | 47.2        | 10.07 iter/s | [Google](https://drive.google.com/file/d/1nL2KYqxSnSZH8c7PRr_d9KEFxCEiyjAR/view?usp=sharing) [Baidu(code: jkyc)](https://pan.baidu.com/s/1RdldnKY93xsh0eWzz8nmgg) | [cfg](../configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DUC)](../configs/halpe_136/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | Symmetric Integral | 45.1        | 16.17 iter/s | [Google](https://drive.google.com/file/d/1zZotfE3WsBe1BxKimlK56wwJuK9E4EDs/view?usp=sharing) [Baidu(code: gaxj)](https://pan.baidu.com/s/1Tm_pV88kFkfqmw2Rzov8xg) | [cfg](../configs/halpe_136/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    |

For example, you can run with:
```
python scripts/demo_inference.py --cfg configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/halpe136_fast50_regression_256x192.pth --indir examples/demo/ --save_img
```

#### Notes
- All of above models are trained only on Halpe Full-body dataset.
- The APs are tested under Halpe's criterion, with flip test on.
- Combined loss means we use heatmap loss (mse loss) on body and foot keypoints and use symmetric integral loss (l1 joint regression loss) on face and hand keypoints.
- There are two FastPose-DCN models with combined loss. The second one uses ten times of weight of hand keypoints, so it is more accurate on hand keypoints but less accurate on the other keypoints.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.

## [COCO WholeBody dataset](https://github.com/jin-s13/COCO-WholeBody) (133 keypoints)

| Model                    | Backbone | Detector | Input Size | Loss Type |     AP     | Speed |  Download | Config |
|--------------------------|----------|----------|------------|------------|------------|-------|-----------|--------|
|[Fast Pose](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    | ResNet50           | YOLOv3 | 256x192            | Symmetric Integral | 55.4      | 17.42 iter/s | [Google](https://drive.google.com/file/d/1WQlwRw7KiKBI2Wyb-lvnQX29R29NbhLz/view?usp=sharing) [Baidu(code: nw03)](https://pan.baidu.com/s/1I1yXJXgKQEag5IUhc3xFGQ) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-regression.yaml)    |
|[Fast Pose (DCN)](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Symmetric Integral | 57.7  | 16.70 iter/s | [Google](https://drive.google.com/file/d/10MgWM4rMORVaHNPyswal7RtrsehVV79X/view?usp=sharing) [Baidu(code: dq9k)](https://pan.baidu.com/s/1cz6lB-xIuwzBBFc1d7p67A) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    |
|[Fast Pose](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    | ResNet50           | YOLOv3 | 256x192            | Combined | 57.8       | 10.28 iter/s | [Google](https://drive.google.com/file/d/14wrc9q96bYqUc2efT8p8XzdTvLm-LwUT/view?usp=sharing) [Baidu(code: 7a56)](https://pan.baidu.com/s/1nML2nHn91-9n5B59axeYwA) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-combined.yaml)    |
|[Fast Pose (DCN)](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    | ResNet50 - dcn           | YOLOv3 | 256x192            | Combined | 58.2       | 10.22 iter/s | [Google](https://drive.google.com/file/d/1aP0nYujw32H-VoJBVsXS-DsBBY-UwI8Y/view?usp=sharing) [Baidu(code: 99ee)](https://pan.baidu.com/s/1dbY6rELFy-ZTJptN5fsUqg) | [cfg](../configs/coco_wholebody/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml)    |
|[Fast Pose (DUC)](../configs/coco_wholebody/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    | ResNet152           | YOLOv3 | 256x192            | Symmetric Integral | 56.9        | 15.72 iter/s | [Google](https://drive.google.com/file/d/1ktBwkG1KL3_iFbPXAh5gua0zX92p-1KV/view?usp=sharing) [Baidu(code: jw3u)](https://pan.baidu.com/s/1TSI2JLk0o5lFPwGf216tNg) | [cfg](../configs/coco_wholebody/resnet/256x192_res152_lr1e-3_1x-duc.yaml)    |

#### Notes
- All of above models are trained only on COCO WholeBody dataset.
- The APs are tested under COCO WholeBody's criterion, with flip test on.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.



#### Notes
- These models are strongly recommended because they are more accurate and flexible.
- These models are trained with multi-domain knowledge distillation (MDKD, see our [paper]() for more details).
- The APs are tested under Halpe's criterion, with flip test on.
- If you want to use the single hand model, you should give the rough bounding box of **a single hand** instead of that of a whole person.
- The speed is tested on COCO val2017 on a single NVIDIA GeForce RTX 3090 gpu, with `batch_size=64` in each iteration and offline yolov3 human detection results.

## 3D Human Pose & Shape Estimation

| Model                    | Backbone | Input Size |     PA-MPJPE (3DPW)     |     PA-MPJPE (Human3.6M)     |  Download | Config |  
|--------------------------|----------|------------|------------|------------|-----------|--------|
|[HybrIK](../configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml)    | ResNet34           | 256x256            |  45.3        |  36.3      | [model](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) | [cfg](../configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml)    |

   </details>
   
## Quick Start
- **Colab**: We provide a [colab example](https://colab.research.google.com/drive/1_3Wxi4H3QGVC28snL3rHIoeMAwI2otMR?usp=sharing) for your quick start.

- **Inference**: Inference demo
``` bash
./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```

Inference SMPL (Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) and put it in `model_files/`).
``` bash
./scripts/inference_3d.sh ./configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml ${CHECKPOINT} ${VIDEO_NAME} # ${OUTPUT_DIR}, optional
```
For high level API, please refer to `./scripts/demo_api.py`. To enable tracking, please refer to [this page](./trackers).

- **Training**: Train from scratch
``` bash
./scripts/train.sh ${CONFIG} ${EXP_ID}
```

- **Validation**: Validate your model on MSCOCO val2017
``` bash
./scripts/validate.sh ${CONFIG} ${CHECKPOINT}
```

Examples:

Demo using `FastPose` model.
``` bash
./scripts/inference.sh configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml pretrained_models/fast_res50_256x192.pth ${VIDEO_NAME}
#or
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
#or if you want to use yolox-x as the detector
python scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/
```

Train `FastPose` on mscoco dataset.
``` bash
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

More detailed inference options and examples, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md)


## Common issue & FAQ
Check out [faq.md](docs/faq.md) for faq. If it can not solve your problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!

## Contributors
AlphaPose is based on RMPE(ICCV'17), authored by [Hao-Shu Fang](https://fang-haoshu.github.io/), Shuqin Xie, [Yu-Wing Tai](https://scholar.google.com/citations?user=nFhLmFkAAAAJ&hl=en) and [Cewu Lu](http://www.mvig.org/), [Cewu Lu](http://mvig.sjtu.edu.cn/) is the corresponding author. Currently, it is maintained by [Jiefeng Li\*](http://jeff-leaf.site/), [Hao-shu Fang\*](https://fang-haoshu.github.io/),  [Haoyi Zhu](https://github.com/HaoyiZhu), [Yuliang Xiu](http://xiuyuliang.cn/about/) and [Chao Xu](http://www.isdas.cn/). 

The main contributors are listed in [doc/contributors.md](docs/contributors.md).

## TODO
- [x] Multi-GPU/CPU inference
- [x] 3D pose
- [x] add tracking flag
- [ ] PyTorch C++ version
- [x] Add model trained on mixture dataset (Check the model zoo)
- [ ] dense support
- [x] small box easy filter
- [x] Crowdpose support
- [ ] Speed up PoseFlow
- [x] Add stronger/light detectors (yolox is now supported)
- [x] High level API (check the scripts/demo_api.py)

We would really appreciate if you can offer any help and be the [contributor](docs/contributors.md) of AlphaPose.


## Citation
Please cite these papers in your publications if it helps your research:

    @article{alphapose,
      author = {Fang, Hao-Shu and Li, Jiefeng and Tang, Hongyang and Xu, Chao and Zhu, Haoyi and Xiu, Yuliang and Li, Yong-Lu and Lu, Cewu},
      journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title = {AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time},
      year = {2022}
    }
    
    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @inproceedings{li2019crowdpose,
        title={Crowdpose: Efficient crowded scenes pose estimation and a new benchmark},
        author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
        pages={10863--10872},
        year={2019}
    }

If you used the 3D mesh reconstruction module, please also cite:

    @inproceedings{li2021hybrik,
        title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3383--3393},
        year={2021}
    }

If you used the PoseFlow tracking module, please also cite:

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }





## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at mvig.alphapose[at]gmail[dot]com and cc lucewu[[at]sjtu[dot]edu[dot]cn. We will send the detail agreement to you.
