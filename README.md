# Iris Classification Project


This project centers on developing a data processing script for the Iris flower dataset from Wikipedia. The script uploads the dataset, splits it into training and inference sets, and saves them locally. The project's goal is to train a classification model using PyTorch library, conducting the entire training process within a Docker container.
## Prerequisites


### Setting Up Development Environment

- For this project I used PyCharm as Integrated Development Environment (IDE) which can be installed from [here](https://www.jetbrains.com/help/pycharm/installation-guide.html).

- Python version 3.10.12


### Installing Docker Desktop

Install Docker Desktop for your operating system from the official [Docker Download Page](https://www.docker.com/products/docker-desktop). 

### Install Dependencies

Add all the necessary dependencies and libraries to the `requirements.txt` file. Include specific versions if needed.
```
pip install -r requirements.txt
```

## Project Structure

```
hw8
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_inference_data.csv
│   └── iris_train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```


## Settings

Manage project configurations with `settings.json`. Update paths and parameters as needed. Create a `.env` file or set an environment variable `CONF_PATH=settings.json` for script execution.

## Data

Generate data using `data_process/data_generation.py`. This script separates data generation from other concerns.

## Training

The training script is in `training/train.py`. To train using Docker:

```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
docker run -it training_image /bin/bash
```

To run locally: 
```
python3 training/train.py
```

## Inference

Use inference/run.py for inference. To run using Docker:
```
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
docker run -v /path_to_local_model:/app/models -v /path_to_input:/app/input -v /path_to_output:/app/output inference_image
```
To run locally: 
```
python inference/run.py
```