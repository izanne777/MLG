# Multivariate Time Series Forecasting Model with Graph Neural Networks Incorporating Multi-scale Local and Global Information Fusion

by Guan-Hua Chen,  Kai-Lung Hua

## Requirements
Recommended version of OS & Python:
* OS: Ubuntu 18.04.2 LTS
* Python: python3.7 (instructions to install python3.7).
```
pip install --upgrade pip
pip install -r requirements.txt
```
## Data Preparation
### Multivariate time series datasets
Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.

### Traffic datasets
Download the METR-LA dataset from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX/) provided by Li et al. 

```
mkdir -p data/METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
```


