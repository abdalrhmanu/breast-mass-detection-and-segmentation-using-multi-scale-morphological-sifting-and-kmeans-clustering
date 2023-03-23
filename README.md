# <h1 align="center">Mammogram Mass Detection</h1>


This repository holds the source code for the project of mammogram mass detection. All source code is kept private until the project is finalized, where this README file will be updated constantly.

Setup Environment Locally
============

To set up a virtual environment, follow the procedure found in <a href="https://github.com/abdalrhmanu/mammogram-mass-detection/blob/main/env.setup.md" target="_blank"> `env.setup.md`</a>.

Directory Structure
============

```
.
├── dataset             # Holds the project dataset folders and files.
    ├── groundtruth     # Ground truth dataset files.
    ├── images          # Dataset images files.
    ├── masks           # Dataset masks files.
    ├── overlay         # Dataset overlay files.
    ├── all.txt         # All images names stored in a .txt file.
    ├── negatives.txt   # All negative labelled images names stored in a .txt file.
    └── positives.txt   # All positive labelled images names stored in a .txt file.
├── literature          # Documentation/paper/project description, etc..
├── helpers             # Some developed packages and modules.
└── notebooks           # Jupyter notebooks used for development.

```

Dataset Setup
============

The dataset folder structure is uploaded in this repository without any of the images. Thus, the dataset folder (from project drive) has to be downloaded and the folders inside has to be moved to this repository folder (inside the dataset root folder). 
