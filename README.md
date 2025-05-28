# NYCU Computer Vision 2025 Spring HW4
StudentID: 313551078 \
Name: 吳年茵

## Introduction
In this lab, we have to train an __image restoration__ model to solve an __image restoration task__ to let the two types of degraded images, Rain and Snow, be clean.

We use a dataset of 1600 degraded images per type for training and validation, and 50 for testing. For this lab, I split the data into a ratio of 8:2 for training and validation. 
I adopted the __PromptIR__ model from the official public GitHub as my base model.

To improve the model's performance for this task, I tried to modify the loss function and added some tricks, such as introduce module or modify the architecture, to have better metrics in Peak Signal-to-Noise Ratio(PSNR) that computes the difference of two images via the concept of mean square error.


## How to install
1. Clone this repository and navigate to folder
```shell
git clone https://github.com/nianyinwu/CV_HW4.git
cd CV_HW4
```
2. Install environment
```shell
conda env create --file hw4.yml --force
conda activate hw4
pip install scikit-image
pip install torchmetrics
pip install einops

```

3. Dataset
```shell
Create a folder named datas and put decompress data to this folder
Rename the test-release folder to test
```

## Split the dataset
(need to modify data path in split_data.py)
```shell
cd codes
python3 split_data.py 
```


## Training
```shell
cd codes
python3 train.py -e <epochs> -b <batch size> -lr <learning rate> -d <data path> -s <save path> 
```
## Testing ( Inference )
The predicted results (pred.npz) will be saved in the argument of save path.
```shell
cd codes
python3 inference.py -d <data path> -w <the path of model checkpoints> -s <save path>
```

## Performance snapshot
![image](https://github.com/nianyinwu/CV_HW4/blob/main/result/snapshot.png)
