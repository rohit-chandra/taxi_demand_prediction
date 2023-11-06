## Taxi Demand Predictor Service 🚕

- This repo is aimed at making it **easy to start playing and learning** about **MLOps**. 

- My interest in creating this project was ignited after reading UBER's blog post on (:link: [Demand and ETR Forecasting at Airports](https://www.uber.com/en-GB/blog/demand-and-etr-forecasting-at-airports/))

## Table of Contents 📑
  * [Quick Setup](#quick-setup)
  * [Problem Statement](#problem-statement)
  * [Data Processing](#data-processing)
  * [Model Training](#model-training)
  * [MLOps](#mlops)
  * [Live Demo](#live-demo)


## Quick Setup

1. Install [Python Poetry](https://python-poetry.org/)
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. cd into the project folder and run
    ```
    $ poetry install
    ```

3. Activate the virtual env that you just created with
    ```
    $ poetry shell
    ```

## Problem Statement

- You work as a data scientist 👨‍🔬👩‍🔬 in a ride-sharing app company 🚗 (e.g. Uber)

- Your job is to help the operations team **keep the fleet as busy as possible**.

### Supply 🚕 and demand 👨‍💼

<p align="left">
<img src="readme_pics/supply_demand_1.PNG"/>
</p>

<p align="left">
<img src="readme_pics/supply_demand_2.PNG"/>
</p>

<p align="left">
<img src="readme_pics/supply_demand_3.PNG"/>
</p>


## Data Processing

**Step 1 - Data Validation** ✔️ ❎

<p align="left">
<img src="readme_pics/step1.PNG"/>
</p>


**Step 2 - Raw data into time-series data**

<p align="left">
<img src="readme_pics/step2.PNG"/>
</p>


**Step 3 - Time-series data into (features, target) data**

<p align="left">
<img src="readme_pics/step2.PNG"/>
</p>

**Step 4 - From raw data to training data**

<p align="left">
<img src="readme_pics/step2.PNG"/>
</p>

**Step 5 - Explore and visualize the final dataset**

<p align="left">
<img src="readme_pics/step5.PNG"/>
</p>



## Model training

<p align="left">
<img src="readme_pics/howtobuildgoodml.PNG"/>
</p>

## MLOps

### Batch-scoring system 🤹

- It is a sequence of steps of computing and storage that map recent data to predictions that can be used by the business

**Step 1 - Prepare data**

- First pipeline - `Data Preparation pipeline or Feature pipeline` - This component runs every hour
- For eg: every hour, we extract raw data from an external service - from a data warehouse or wherever the recent data is
- Once we fetch raw data, we then create a tabular dataset with features and target and store them in the feature store
- This is the Data Ingestion Pipeline
  
<p align="left">
<img src="readme_pics/preparedata.PNG"/>
</p>

**Step 2 - Train ML Model**
- 2nd pipeline - `Model Training pipeline`
- Retrain the model since ML models in real-world systems are trained regularly
- In this project, It's on-demand, whenever I think I want to train the model, I can trigger this pipeline, and it automatically trains, generate a new model and save it back to the model registry
  
<p align="left">
<img src="readme_pics/trainmlmodel.PNG"/>
</p>

**Step 3 - Generate predictions on recent data**
- 3rd pipeline - `Prediction pipeline`
- USe most recent features and current model we have in production to generate predictions
<p align="left">
<img src="readme_pics/generatepredictions.PNG"/>
</p>


**Serverless MLOps tools**

- **`Hopsworks`** as our feature store
   - It's a serverless platform that provides an infrastructure  to manage and run the feature store automatically
   - It's easy to manage unlike GCP, Azure where we have to setup different components first

- **`Github Actions`** to schedule and run jobs
   - We automate the feature pipeline that will ingest data every hour
   - The notebook is going to automatically run every hour and it's going to fetch a batch of recent data, transform it and save it into features store
   - Created a configuration yaml file under `.github/workflows`
   - The cron job runs every hour
   - The command below triggers the notebook execution from command line

```
poetry run jupyter nbconvert -to notebook -- execute notebooks/12_feature_pipeline.ipynb
```

**Feature Store**
- Feature store is used to store features.
- These features can be used to either train the models or make predictions.
- Features saved in the feature store are:
   - `pickup_hour`
   - `no_of_rides`
   - `pickup_location_id`
     
<p align="left">
<img src="readme_pics/featurestore1.PNG"/>
</p>


**Backfill the Feature Store**
- Fetch files from the year 2022
- Transform raw data into time series data
- Dump it in the feature store
- Repeat for the year 20223 and so on

## Live Demo
 - work in progress
