# Neural-Networks-Project

Git large file storage is used to track the two datasets, 
- Arrhythmia_Dataset/*
- Heartbeat_Dataset/*

```bash
python -m venv .venv
. .venv/Scripts/activate 
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Add CUDA to environment file path
```bash
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp"

$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\cudnn\bin"

```
## ACG Arrhythmia Classification Data

Brief details of the dataset are as follows:

There are four ECG arrhythmia datasets in here, each employing 2-lead ECG features. Datasets obtained from PhysioNet are MIT-BIH Supraventricular Arrhythmia Database, MIT-BIH Arrhythmia Database, St Petersburg INCART 12-lead Arrhythmia Database, and Sudden Cardiac Death Holter Database.


1. In each of the datasets, the first column, named "record" is the name of the subject/patient.

2. Each data contain five classes/categories: N (Normal), S (Supraventricular ectopic beat), V (Ventricular ectopic beat), F (Fusion beat), and Q (Unknown beat). The column "type" contains the class information.

3. The remaining 34 columns contain 17 features for each ECG lead (17 features for lead-II and 17 features for lead-V5)

4. The short description of the features for all four datasets can be found in the following figure. Note that the features are consistent across all four datasets to be comparable when applying Machine Learning models.


| Feature Group                      | Lead A and B            |
|------------------------------------|--------------------------|
| **RR Intervals**                   |                          |
|                                    | Average RR               |
|                                    | RR                       |
|                                    | Post RR                  |
| **Heartbeat Intervals Features**   |                          |
|                                    | PQ Interval              |
|                                    | QT Interval              |
|                                    | ST Interval              |
|                                    | QRS Duration             |
| **Heart Beats Amplitude Features** |                          |
|                                    | P peak                   |
|                                    | T peak                   |
|                                    | R peak                   |
|                                    | S peak                   |
|                                    | Q peak                   |
| **Morphology Features**            |                          |
|                                    | QRS morph feature 0      |
|                                    | QRS morph feature 1      |
|                                    | QRS morph feature 2      |
|                                    | QRS morph feature 3      |
|                                    | QRS morph feature 4      |
