# bachelorproef

Github repository can be found [here](https://github.com/ilianbronchart/bachelorproef)
Tobii Glasses 3 API Documentation can be found [here](https://tobiipro.github.io/g3pylib/g3pylib.html)

## Installing

### Prerequisites

All development was done on Windows Subsystem for Linux (WSL) with Ubuntu 22.04.5

#### Poetry

Poetry is a python dependency manager, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation)
During development, version `2.0.1` was used.

> Note: If you are using WSL, make sure to set `poetry config virtualenvs.create true` to create a dedicated venv folder for the project.
> This will ensure that your IDE recognizes the virtual environment and can use it for code completion and linting. 

#### Just

Just is a command runner that allows you to run commands defined in a `justfile`. It is similar to Make, but with a simpler syntax.
```bash
sudo apt install just
```

#### Docker

Install docker by following the instructions [here](https://docs.docker.com/engine/install/)

Install Nvidia Container Toolkit by following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-dnf-rhel-centos-fedora-amazon-linux)
This is required to run the application in docker with GPU support.

#### Cuda

This application was developed using CUDA 12.8 (see `Dockerfile`).
If you want to run the application outside of docker, you need to install CUDA on your host machine.

#### TensorRT

While TensorRT is not used in the application, the Dockerfile expects the TensorRT libraries to be present in the `libs` folder.
You can download the TensorRT tar file by following the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#download). The version that was used during development can be found in the `Dockerfile`.
Then, put the tar file in the `libs` folder.

### Python Requirements and Model Checkpoints

The justfile contains an install command that will install all the required dependencies and download the necessary libraries.
It also downloads the required SAM2 checkpoints.
Use the following command to install the dependencies:
```
just install
```

Also, download the FastSAM checkpoints by following the Ultralytics instructions [here](https://docs.ultralytics.com/models/fast-sam/#installation) and place them in the `checkpoints` folder.
The application expects the `FastSAM-x.pt` to be present (even though it currently does not use it).
See `src/config.py` for settings and expected files.

## Running the Application

Run the following command to build the docker image:
```bash
just build
```
> Note: This sets the CUDA and TensorRT versions as specified at the top of the Justfile. If you want to use a different version, you can change the `CUDA_USER_VERSION` and `TENSORRT_VERSION` variables in the Justfile.

The repository contains a `docker-compose-dev.yml` file that can be used to run the application in development mode.
It binds the following files and folders to the container:
- `./data`: The folder where the application data is stored. If this folder does not exist, it will be created automatically.
- `./src`: The folder where the application source code is stored. When saving changes to the source code, the container will automatically reload the application.
- `./database.db`: The SQLite database file. If this file does not exist, it will be created automatically.
- `./checkpoints`: The folder where the SAM2 checkpoints are stored. If this folder does not exist, it will be created automatically.

Run the following command to start the application in development mode:
```bash
docker compose -f docker-compose-dev.yml up
```

If you want to run experiments with the application where these files have a different location, you can create your own docker-compose file in the root of the repository based on the `docker-compose-dev.yml` file and change the volume bindings accordingly.


