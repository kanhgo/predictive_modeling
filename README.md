[![App and Notebook CI](https://github.com/kanhgo/predictive_modeling/actions/workflows/main.yml/badge.svg)](https://github.com/kanhgo/predictive_modeling/actions/workflows/main.yml)

# EMPOWERING ACTION FOR BETTER HEALTH OUTCOMES
# Objective
To develop a predictive model to forecast the likelihood of diabetes diagnosis using demographic and behavioral risk factors from the CDCs BRFSS 2015 dataset.

## Importance

Diabetes is a significant public health concern in the United States, affecting millions of individuals across various demographics. By using predictive analytics and data-driven approaches, health planners can identify high-risk populations and allocate resources more effectively. For instance, models can guide the implementation of educational programs in schools to break the normalization of diabetes and its prevalence. Additionally, these models can help in planning initiatives to increase access to fresh food options in neighborhoods, thus promoting healthier lifestyles and potentially reducing the incidence of diabetes and improving health outcomes overall.

Althought this dataset and modeling exercise is US focused, it also provides a reference point for efforts in other geographical locations where diabetes prevelance is a concern. An understanding of cultural differences and practices can guide creation of an appropriate survey to curate and collect relevant data to support predictive modeling.

## About the dataset
- Data source: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). This data subset was based on the CDC Behavioral Risk Factor Surveillance System (BRFSS)(2015) data found [here](https://www.cdc.gov/brfss/annual_data/annual_2015.html). 
- Obtained clean and composed of 253,680 rows of survey responses, with 21 feature variables.
- The feature variables are predominantly categorical (including numeric ranges such as 5-year age ranges, or income ranges), with the classes represented by nominals e.g. 0, 1 (where the numbers are just labels and don't represent any order). BMI, number of days mental health not good (MentHlth), and number of days physical health not good (PhysHlth) are the continuous variables present.
- The target variable has three (3) classes which have also been represented by nominals: 0 - no diabetes or only gestational, 1 - prediabetes, 2 - diabetes.
