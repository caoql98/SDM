# Scale-aware Detailed Matching for Few-Shot Aerial Image Semantic Segmentation
Codes and dataset (iSAID-5i) for Scale-aware Detailed Matching for Few-Shot Aerial Image Semantic Segmentation, and the work has been accepted by TGRS


the overall network:
<p align="left">
<img src="img/remote_sensing1.png" alt="the overall network" width="700px">
</p>
some visualization results:
the overall network:
<p align="left">
<img src="img/remote_sensing_result.png" alt="the results" width="800px">
</p>


### Training
```
cd scripts
sh train_group0.sh
```
### Inference
If you want to test all of the saved models, you can use:
```
python test_all_frame.py
```
### Environment
+ python == 3.7
+ pytorch1.0

+ torchvision,
+ pillow,
+ opencv-python,
+ pandas,
+ matplotlib,
+ scikit-image




### Datasets and Data Preparation

The newly provied dataset [**iSAID-5i**](https://pan.baidu.com/s/1kGvYMkHoV1eBM1k4VSG-HA)        
(Password:nwpu)
or  [**iSAID-5i**](https://drive.google.com/file/d/17PQ1iKCbaj2OjwBdCn_VBh09ntI4lxgL/view?usp=sharing)

### BibTex
```BibTex
@article{yao2021scale,
  title={Scale-aware detailed matching for few-shot aerial image semantic segmentation},
  author={Yao, Xiwen and Cao, Qinglong and Feng, Xiaoxu and Cheng, Gong and Han, Junwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--11},
  year={2021},
  publisher={IEEE}
}
