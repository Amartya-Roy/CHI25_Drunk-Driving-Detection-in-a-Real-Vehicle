# CHI2025 - Moving Beyond the Simulator: Interaction-Based Drunk Driving Detection in a Real Vehicle Using Driver Monitoring Cameras and Real-Time Vehicle Data
This repository contains the source code for the preprocessing, training, and evaluation pipelines used for our manuscript submitted to CHI 2025:


>Interaction-Based Drunk Driving Detection in a Real Vehicle Using Driver Monitoring Cameras and Real-Time Vehicle Data

## Content overview
The repo consists of three majors parts, which are subdivided into the separate folders of this repository as follows:

1. 01_eye_tracking_preprocessing: Contains the preprocessing and extraction of features from eye tracking data.
2. 02_can_data_preprocessing: Contains the preprocessing and extraction of features from the CAN bus data.
3. 03_train_and_predict: Contains the code for training and evaluating the models based on the features extracted from the eyetracking and the CAN bus data.

## How to use the code
The provided code can be used to build a model for drunk driving detection based on driver monitoring camera and CAN bus data. To use the code, you have to follow the following three steps below **in order**.

>**Prerequisites:** We recommend to use Python >= 3.9.

### 1. Preprocessing eye tracking data
The first step is to preprocess the eye tracking data to extract and aggregate eye tracking features. 
#### Adapting the config_preprocessing.yml
Inside the folder 01_eye_tracking_preprocessing, you can find the [config_processing.yml](01_eye_tracking_preprocessing/config_processing.yml). It contains some settings, such as the selected probands and the folders for the raw data and processing output. Change the following two lines you require, to load your own data:
```yaml
raw_input_directory: '/test_track'
preprocessed_output_directory: '/test_track_processed/'
```

#### Adapting the config_aggregation.yml
Inside the folder 01_eye_tracking_preprocessing, you can find the [config_aggregation.yml](01_eye_tracking_preprocessing/config_aggregation.yml). It contains the selected probands, directories and some settings for the merge step. Adjust the path and any other options as needed. ***Please use the same path that was used for preprocessing!***

```yaml
data_directory_processed: '/test_track_processed'
```



#### Installing the requirements
Please install the python requirements using the command below:

```sh
pip install -r 01_eye_tracking_preprocessing/requirements.txt
```

#### Running the code for processing and aggregating eye tracking data
First, change the directory of your terminal to the eye tracking subfolder:
```sh
cd 01_eye_tracking_preprocessing
```

Then, run the processing:
```sh
python 01_run_processing.py
```
Finally, run the aggregation to merge the processed files into a single dataset:
```sh
python 02_run_aggregation.py
```

### 2. Preprocessing CAN data
The second step is to preprocess the CAN bus data to extract and aggregate features, such as brake and acceleration behavior and steering wheel movement.

#### Adapting the config_processing.yml
Just as for the eyetracking data, you have to adopt the configs for the CAN data to specify the paths to your raw data. 

Inside the folder 02_can_data_preprocessing, you can find the [config_processing.yml](02_can_data_preprocessing/config_processing.yml). Specify the paths for the data_directory and the data_output_directory accordingly. **Make sure to use the same paths as used for the eyetracking data!**
```yaml
data_directory: '/test_track'
data_output_directory: '/test_track_processed'
```

#### Adapting the config_aggregation.yml
Inside the folder 02_can_data_preprocessing, you can find the [config_aggregation.yml](02_can_data_preprocessing/config_aggregation.yml). Specify the paths for the data_directory and the data_output_directory accordingly. **Make sure to use the same paths as the output path used for the eyetracking data!** Also, **add the prefix "/canlogger/" to the data_output_directory!**
```yaml
data_directory: '/test_track_processed'
relative_subject_output_directory: 'canlogger' # need to be same as the one in config_processing.yml
data_output_directory: '/test_track_processed/canlogger/'
```

#### Installing the requirements
Install the python requirements using the command below:
```sh
pip install -r 02_can_data_preprocessing/requirements.txt
```


#### Running the code for processing and aggregating CAN data
First, change the directory of your terminal to the can_data subfolder:
```sh
cd 02_can_data_preprocessing
```

Then, run the processing:
```sh
python 01_run_processing.py
```
Finally, run the aggregation to extract the features from the processed data:
```sh
python 02_run_aggregation.py
```

### 3. Training and evaluating the prediction model
Once the eye tracking and CAN data is processed and aggregated, you can use it to train the prediction model. Follow the instructions below.

#### Adapting the config_processing.yml
Inside the folder 03_train_and_predict, you can find the [config_prediction.yml](03_train_and_predict/config_prediction.yml). 
Change the **data_directory** to point to the **processed** data (i.e., the output of the eyetracking and CAN preprocessing).
```yaml
data_directory: 'test_track_processed'
```

#### Installing the requirements
Install the python requirements using the command below:
```sh
pip install -r 03_train_and_predict/requirements.txt
```

#### Running the code for training and evaluating the model
First, change the directory of your terminal to the eye tracking subfolder:
```sh
cd 03_train_and_predict
```

Then, run the training and evaluation:
```sh
python run_train_and_eval.py
```


## Contributors
Contributors currently hidden due to ongoing peer review process.

## Contact
Contributors currently hidden due to ongoing peer review process.

## Citation
Please cite our work as follows: TBA

