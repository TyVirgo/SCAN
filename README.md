# SCAN

## Dependencies

- The cuda version used in the project is 12.2
- Python = 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
```
conda create -n your_env_name python=3.8
conda activate your_env_name
```
- Other packages required by the project are in the file ‘requirements.txt’
```
pip install -r requirements.txt
```

# Codes 
- This codes provide the testing and training code.
  
## How to Test
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing)

2. Three factor of pretrained models have be placed in `./weights/` folder. 

3. The testing commands are placed in the './src/demo.sh' file. 

4. The output results will be sorted in `./experiment/test/`

## How to Train

1. 下载 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from [Here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) 

3. Modify the training dataset path attributes '--dir_data' and '--data_train' in the `./src/option.py` file.

4. The training commands are placed in the './src/demo.sh' file.
Close comments in 'demo.sh' and run 'demo.sh' to execute the corresponding command of training. Such as:
```
python main.py --model XXX 
```
4. The training results will be sorted in `./experiment/train/`

## SR images visualization
1. We provided visualization of SR images for SCAN from [Google Drive](https://drive.google.com/XXX) or [Baidu Drive](https://pan.baidu.com/XXX) (code：s8eg)

## Contact
Email: fjs1867@mnnu.edu.cn


If you find our work is useful, please kindly cite it.（需要换自己文章的引用）
```
cite code
```

## License
This project is released under the Apache 2.0 license.


## Acknowledgements
This code is built on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks for the awesome work.
