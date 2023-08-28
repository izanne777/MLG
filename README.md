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
Download Solar-Energy, Electricity datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.

### Traffic datasets
Download the METR-LA dataset from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX/) provided by Li et al. 

```
mkdir -p data/METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
```

## Model Training
### Solar-Energy
```
python train_single_step.py --save ./model-solar-3.pt --data ./data/solar_AL.txt --num_nodes 137 --layers 4 --conv_kernel [6, 12] --hid_size 6 --batch_size 4 --epochs 30 --horizon 3
```
### Electricity
```
python train_single_step.py --save ./model-electricity-3.pt --data ./data/electricity.txt --num_nodes 321 --layers 5 --conv_kernel [12, 16] --hid_size 6 --batch_size 4 --epochs 30 --horizon 3
```
### METR-LA
```
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207 --conv_kernel [12, 16] --hid_size 6 --batch_size 4 --epochs 30 --horizon 3
```

## Our Baseline
[Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2005.11650.pdf)

## Conference
Under Review
