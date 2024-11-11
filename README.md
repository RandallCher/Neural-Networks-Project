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

# ECG Heartbeat Categorization Dataset

## Abstract
This dataset is composed of two collections of heartbeat signals derived from two famous datasets in heartbeat classification, the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. The number of samples in both collections is large enough for training a deep neural network.

This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing some of the capabilities of transfer learning on it. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals are preprocessed and segmented, with each segment corresponding to a heartbeat.

## Content
Arrhythmia Dataset
Number of Samples: 109446
Number of Categories: 5
Sampling Frequency: 125Hz
Data Source: Physionet's MIT-BIH Arrhythmia Dataset
Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
The PTB Diagnostic ECG Database
Number of Samples: 14552
Number of Categories: 2
Sampling Frequency: 125Hz
Data Source: Physionet's PTB Diagnostic Database
Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.

## Data Files
This dataset consists of a series of CSV files. Each of these CSV files contain a matrix, with each row representing an example in that portion of the dataset. The final element of each row denotes the class to which that example belongs.