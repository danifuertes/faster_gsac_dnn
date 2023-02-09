## Faster GSAC-DNN
Faster GSAC-DNN (Grid of Spatial Aware Classifiers based on Deep Neural Networks) is an evolution of
[GSAC](https://www.sciencedirect.com/science/article/abs/pii/S0923596521000023) and
[GSAC-DNN](https://www.sciencedirect.com/science/article/pii/S1051200422000902?via%3Dihub) with faster and more
efficient performance. For more details, please see our paper. If this repository is useful for your work, please cite
our paper:

```

``` 

## Software requeriments

This code has been tested on Ubuntu 18.04.6 LTS with Docker 20.10.12, Python 3.8, TensorFlow 2.9.1, CUDA 11.6 and two
Nvidia TITAN Xp GPUs. The dependencies can be obtained as follows:

1. Build the Docker image with `docker build -t fgd_image .`
2. Run the Docker container with `docker run --user $(id -u):$(id -g) --gpus all -it --rm --volume=$(pwd):/home/app:rw --volume=/path/to/dataset:/path/to/dataset:ro --name fgd_container fgd_image bash`

This code can also be executed without docker:

1. Install Python 3.8 with `sudo apt install python3.8`. Other versions may also be valid.
2. Create a virtual environment with `virtualenv --python=python3.8 venv`.
3. Activate the virtual environment with `source venv/bin/activate`.
4. Install the latest drivers for your GPU.
5. Install CUDA 11.6 following [these instruccions](https://developer.nvidia.com/cuda-downloads)
6. Install cuDNN 8.5 following [these instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)
7. Install the dependencies with `pip3 install -r requirements.txt`.

## Dataset & Training

Your dataset should be divided on 2 main sets: train and test. The structure of your dataset should be similar to the
following one:
```
/DatasetName
    train.txt
    test.txt
    /train
        /train_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /train_sequence_2
          ...
        /train_sequence_N
          ...
    /test
        /test_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /test_sequence_2
          ...
        /test_sequence_N
          ...
```
The `train.txt` and `test.txt` files should contain a list of the train and test annotations, respectively. The
ground-truth format is described next:
```
/relative/path/to/train_sequence_1/000000000.png x,y
/relative/path/to/train_sequence_1/000000001.png x,y x,y x,y
/relative/path/to/train_sequence_1/000000002.png
/relative/path/to/train_sequence_1/000000000.png x,y x,y
...
/relative/path/to/train_sequence_2/000000000.png x,y
/relative/path/to/train_sequence_2/000000001.png
/relative/path/to/train_sequence_2/000000002.png x,y x,y x,y x,y
...
```
where x and y are the coordinates of each of the point-based annotations on the image. You can configure your validation
data like your train and test data. The file with the annotations of your validation set should be called `val.txt`. If
your dataset does not contain a validation set, you can provide a percentage with the option `--val_perc` to extract
some random samples from the training set and use them to validate:

```bash
python train.py --dataset_path /path/to/your/dataset --dataset_name DatasetName --val_perc 0.1 --input_dim 224 224 3 --grid_dim 20 20
```

In case you have a validation set with a format similar to the one described above, you can train your model with:

```bash
python train.py --dataset_path /path/to/your/dataset --dataset_name DatasetName --input_dim 224 224 3 --grid_dim 20 20
```

While you are training a model, the weights that optimize the validation loss are saved in 
`models/DatasetName/ModelName_TrainDate` by default. To restore a model, you should use the option
`--restore_model True`, indicate the name of the directory of the model with
`--model_dir models/DatasetName/ModelName_TrainDate`, and indicate the complete path to the weights desired with
`--weights weights_XXX.h5`. Example:

```bash
python train.py --restore_model True --model_dir models/DatasetName/ModelName_TrainDate --weights weights_057.h5 --dataset_path /path/to/your/dataset --dataset_name DatasetName --input_dim 224 224 3 --grid_dim 20 20
```

Use multiple GPUs while training with `--multi_gpu True`. Use data augmentation while trining with `--data_aug True`.
Indicate a different backbone with the option `--backbone`. For any additional help, you can run:

```bash
python train.py --help
```

## Test

To evaluate your trained model using your test data with the format described above, you can run:

```bash
python test.py --model_dir models/DatasetName/ModelName_TrainDate --weights weights_057.h5 --dataset_path /path/to/your/dataset --dataset_name DatasetName --dataset_name DatasetName --input_dim 224 224 3 --grid_dim 20 20
```

Note that options related to the structure of the network should not be changed from training to test. In case you do
not remember any of the options, read the file `models/DatasetName/ModelName_TrainDate/log_dir/options.txt`, which
contains a list with the options used to train the model.

To visualize the detections, you can use the option `--visualize True`. Also, you can save the images with the
predictions using the option `--save_imgs True`.

The images are saved by sequences in `models/DatasetName/ModelName_TrainDate/predictions/TestDate/images`. Next to this
directory, you can find 2 directories called `detection-results` and `ground-truth`. These directories contain files
with the predictions and annotations of each test image, respectively. Run the following command to get metrics:

```bash
python map.py --model_dir models/DatasetName/ModelName_TrainDate/predictions/TestDate --img_width 224 --img_height 224
```

Check the folder `models/model_DatasetName_TrainDate/predictions/TestDate/results` to find the results computed. For
any additional help, you can run:

```bash
python test.py --help
python map --help
```

## Note

At the beginning of `train.py` and `test.py` you can set the device changing the value of
`os.environ["CUDA_VISIBLE_DEVICES"]`. An alternative is to type, previous to any python command, the following:

```bash
CUDA_VISIBLE_DEVICES=-1 python ...  # -1 for CPU
CUDA_VISIBLE_DEVICES=0 python ...  # 0 for the 1st GPU
CUDA_VISIBLE_DEVICES=1 python ...  # 1 for the 2nd GPU
...
```

## Acknowledgements
This repository is an adaptation of
[danifuertes/gsac_dnn](https://github.com/danifuertes/gsac_dnn). The ViT backbone was implemented following this
repository: [faustomorales/vit-keras](https://github.com/faustomorales/vit-keras).
