# Bank Failure Prediction

 :speech_balloon:*This document is supplementory for AI final project presentation video. Due to the video time limitation of 15 miniutes, the slides and video can not cover comprehensive experimental details. Hope this attachement can provide more guidence and explaination of our projects.*
 
## Overview
Bank failure refers to financial entities which can not fulfill loan obligation result from liquidity problems. Since financial institutions are usually with high levity and close cash flows exchanges, any default of an institution will cause ripple effect to the whole financial system. Therefore,  the issue of bank failure is more severe than corporate bond default, for its high social loss afterwards.

In this work, we aim to build a bank failure prediction system by implementing data mining classifiers, ensembling learning and KNN, and compared with the baseline models, linear regression and logisitc regression. The datasets include two dimension, financial ratios from FDIC and U.S. macroeconomic indicators collected by ourselves from 2000 to 2020. Since it is a time series problem, we split the dataset into training, from 2000 to 2010, and testing data, after 2011. 

## Dataset
### Definition
The columns of datasets are as below:
Variables|Definition|
-|-|
*Dependent variables*
failure|Binary variable, equals one if the bank fails and zero, otherwise|
*Assets side variables*|	
total_assets|	Bank’s total assets in millions of USD
size|	The natural logarithm of bank’s total assets|
totalinv_assets|Total investments to total assets
htminv_assets|	Total investments in hold-to-maturity securities to total assets
fsale_assets|	Total investments in securities for sale to total assets
refarmcol_assets|	Loan secured by farm land to total assets|
interbank_assets|	Loan to other banks to total assets
loanagri_assets|	Loan to agriculture to total assets|
ciloan_assets|	Commercial and Industrial loan to total assets
creditcard_assets|	Credit card loan to total assets
otherpploan_assets|	Other personal loan to total assets
otherloan_assets|	Total other loan to total assets
*Liabilities side variables*
lev|	Total liabilities to total assets
deposit_assets|	Total deposits to total assets
demanddep_assets|	Demand deposit to total assets
timedep_assets|	Time deposit to total assets
otherresidloan_assets|	Other loan to resident property to total assets
*Assets quality variables*	
npl_assets|	Delinquent loan to total assets
loanlossalwc	Loan loss allowance to total assets|
loanlossprov|	Loan loss provision to total assets
chargeoff_assets|	Total loan charge-offs to total assets
*Bank’s profitability variables*
nim|	Net Interest Margin
nnon_im|	Non-Interest Margin
roa|	Returns on Assets
*Off-balance-sheet activities*
secborrowed_obs|	Securities borrowed to total assets
seclent_obs|	Securities lent to total assets
revolvelines_assets|	Revolving Real Estate line of credit to total assets
creditcard_line_obs|	Credit card line of credit to total assets|
resid_recol_obs|	Resident property line of credit secured by resident property to total assets
resid_norecol_obs|	Resident property line of credit unsecured by resident property to total assets
underwriting_obs|	Underwriting to total assets
derivatives_guarantor_obs|	Derivatives with bank as guarantor to total assets
derivatives_beneficiary_obs|	Derivatives with bank as beneficiary to total assets
*Capital adequacy variables*	
car|	Capital adequacy ratio, calculated as bank capital to Risk-weighted assets
tier1_car|	Tier-1 ratio, calculated as tier-1 capital to Risk-weighted assets
tier2_car|	Tier-2 ratio, calculated as tier-2 capital to Risk-weighted assets
*Macroeconomic indicators*
annual GDP|the monetary value of all finished goods and services made within a country during a specific period
GDP growth|an increase in the total production of economic goods and services, compared from one period to another
housing start|the start of construction on a new residential housing unit
consumer pricing index|the overall change in consumer prices over time based on a representative basket of goods and services
unemployment rate|the share of workers in the labor force who do not currently have a job but are actively looking for work
rate of inflation|the percentage change in the price index for a given period compared to that recorded in a previous period

Descriptive statistics of financial ratios

| Statistics   | failure | size    | htminv_assets | fsale_assets | lev    | refarmcol_assets | interbank_assets | loanagri_assets | ciloan_assets | creditcard_assets | otherpploan_assets | otherloan_assets | demanddep_assets | timedep_assets | otherresidloan_assets | npl_assets | loanlossalwc | loanlossprov | nim    | nnon_im | roa     | chargeoff_assets | secborrowed_obs | revolvelines_assets | creditcard_line_obs | resid_recol_obs | resid_norecol_obs | underwriting_obs | tier1_car |
|--------------|---------|---------|---------------|--------------|--------|------------------|------------------|-----------------|---------------|-------------------|--------------------|------------------|------------------|----------------|-----------------------|------------|--------------|--------------|--------|---------|---------|------------------|-----------------|---------------------|---------------------|-----------------|-------------------|------------------|-----------|
| Observations | 379783  | 379783  | 376006        | 376354       | 372805 | 365882           | 379783           | 357345          | 373522        | 379783            | 353545             | 354190           | 375431           | 354190         | 357345                | 371643     | 360842       | 360712       | 354626 | 360952  | 361026  | 357344           | 357330          | 357345              | 357345              | 354190          | 357345            | 360489           | 355693    |
| Mean         | 0.4426  | 11.8810 | 0.0394        | 0.1805       | 0.8876 | 0.0366           | 0.0002           | 0.0445          | 0.1014        | 0.0002            | 0.0509             | 0.0025           | 0.1102           | 0.6994         | 0.1488                | 0.0018     | 0.0092       | 0.0020       | 0.0230 | -0.0138 | 0.0051  | 0.0019           | 0.0004          | 0.0143              | 0.0049              | 0.0267          | 0.0009            | 0.0000           | 0.1765    |
| Std          | 0.4967  | 1.6158  | 0.0957        | 0.1492       | 0.0921 | 0.0551           | 0.0069           | 0.0796          | 0.0872        | 0.0008            | 0.0495             | 0.0063           | 0.0766           | 0.1234         | 0.1199                | 0.0044     | 0.0070       | 0.0039       | 0.0115 | 0.0095  | 0.0084  | 0.0123           | 0.0094          | 0.0241              | 0.0180              | 0.0414          | 0.0061            | 0.0008           | 0.1563    |
| Minimum      | 0       | 0       | 0             | 0            | 0      | 0                | 0                | 0               | 0             | 0                 | 0                  | 0                | 0                | 0.0154         | 0                     | 0          | 0            | -0.0008      | 0.0047 | -0.0477 | -0.0360 | 0                | 0               | 0                   | 0                   | 0               | 0                 | 0                | 0.0754    |
| p25%         | 0       | 10.8929 | 0             | 0.0651       | 0.8793 | 0                | 0                | 0               | 0.0446        | 0                 | 0.0154             | 0                | 0.0619           | 0.6623         | 0.0620                | 0.0000     | 0.0062       | 0.0001       | 0.0120 | -0.0190 | 0.0023  | 0.0001           | 0               | 0                   | 0                   | 0.0002          | 0                 | 0                | 0.1088    |
| p50%         | 0       | 11.6722 | 0             | 0.1558       | 0.9035 | 0.0102           | 0                | 0.0052          | 0.0810        | 0                 | 0.0377             | 0.0004           | 0.1005           | 0.7207         | 0.1235                | 0.0002     | 0.0082       | 0.0007       | 0.0218 | -0.0126 | 0.0051  | 0.0005           | 0               | 0.0039              | 0                   | 0.0123          | 0                 | 0                | 0.1348    |
| p75%         | 1       | 12.5874 | 0.0272        | 0.2653       | 0.9183 | 0.0534           | 0                | 0.0529          | 0.1321        | 0                 | 0.0700             | 0.0017           | 0.1443           | 0.7696         | 0.2033                | 0.0017     | 0.0107       | 0.0020       | 0.0315 | -0.0070 | 0.0090  | 0.0016           | 0               | 0.0199              | 0.0004              | 0.0353          | 0                 | 0                | 0.1832    |
| Maximum      | 1       | 21.7132 | 1             | 1            | 6.1947 | 0.6494           | 0.9146           | 0.7227          | 0.5066        | 0.0063            | 0.2604             | 0.0427           | 1                | 0.8710         | 0.9825                | 0.2518     | 0.3673       | 0.0261       | 0.0527 | 0.0165  | 0.0302  | 5.9973           | 0.7500          | 0.9140              | 0.1575              | 1.5345          | 0.5043            | 0.1210           | 1.3183    |


### EDA
**Figure 1: Number of financial institutions in years**: Nubmer of banks has a downtrend with year-to-year, which might related to 2008 financial crisis.</br>
<img src="https://i.imgur.com/jaIIaWb.png" width="40%" height="40%">

**Figure 2: Default rate of banks**: Banks are either recidivist of defaulting or never default at all.</br>
<img src="https://i.imgur.com/eN0d1RT.png" width="30%" height="30%">

**Figure 3: Default frequency of banks**:
Most of banks(more than 4000 banks) have no default record at all.</br>
<img src="https://i.imgur.com/4sSdZPW.png" width="40%" height="40%">

## Literature Review
### Factors for bank failure prediction
### Resampling
## Methodology
### Preliminary Investigation: building up baselines
#### In and out-of sample test
In out-of-sample test, the differences of precisions and calls in each models are much larger than in-sample test and the accuracies of logistic and linear regression models can achieve to 80% higher. These phenomena are caused by the imbalanced data that most of data are 0, which is align with models' predictions. However, as for data 1, models cannot predict correctly. Accordingly, in this work, we use macro average, wich is the average of precission and call, to replace accuracy as a evaluation matrix. We take the results in this test as the baselines. The goal of the following work is to balance precision and call and also to improve macro averages.</br>
1. Performance</br>

In-sample|Accuracy|Precision|Call|
-|-|-|-|
XGBRF|0.63|0.74|0.39
RF|1|1|1
XGBoost|0.68|0.76|0.53
KNN|0.76| 0.81| 0.68
Linear| 0.64 |0.75| 0.40
Logistic |0.65 |0.74| 0.48
**Out-of-sample**|**Accuracy**|**Precision**|**Call**|
XGBRF|0.67| 0.79| 0.22
RF|0.73| 0.83 |0.43
XGBoost|0.54| 0.67| 0.25
KNN|0.78 |0.87| 0.43
Linear| 0.86| 0.93| 0.09
Logistic |0.92| 0.92| 0.08

2. In-sample Confusion Matrix

Xgboost|XGBRF|Random Forest
-|-|-
![](https://i.imgur.com/R8GQtvS.png)|![](https://i.imgur.com/KHnV5DP.png)|![](https://i.imgur.com/v0Rbj0N.png)
**KNN**|**Linear regression**|**Logistic regression**
![](https://i.imgur.com/DYBKi9v.png)|![](https://i.imgur.com/iCmJ2Dv.png)|![](https://i.imgur.com/RwHuO7k.png)

3. Out-of-sample Confusion Matrix

Xgboost|XGBRF|Random Forest
-|-|-
![](https://i.imgur.com/9bk2S73.png)|![](https://i.imgur.com/apHcthD.png)|![](https://i.imgur.com/7Yx0DPl.png)
**KNN**|**Linear regression**|**Logistic regression**
![](https://i.imgur.com/LqBsRFG.png)|![](https://i.imgur.com/WPymOZQ.png)|![](https://i.imgur.com/oQWFuI8.png)
### Implementation steps
1. Data preprocessing</br>
- Missing Values
We remove data columns where 95% of values are 0; therefore, interbank_assets, derivatives_guarantor_obs, derivatives_beneficiary_obs are removed. Then, we use iterative imputor to fill in NaN values.<br>
**Figure 4: Correlation Heapmap after Preprocessing**
<img alt="image" src="https://user-images.githubusercontent.com/104308942/173482647-64be2ee5-e811-46d6-96ed-1b913fe1fd7c.png">

- Resampling</br>
Using SmoteTomek and SmoteEnn to help the ratio of possitive and negative data resize to nearly 1.</br>

-|Raw data| SmoteEnn|SmoteTomek
-|-|-|-
Negative(0)|186651|186354|194246
Positive(1)|125957|186562|194454
ratio|1.48|1|1
2. Feature selection
- Filter
- Wrapper
- Wrapper + Filter
- Lasso
3. Scaling and dimension reduction</br>
For quick model convergence, we standardize the data from 0 to 1. For dimension reduction, we use PCA to retain 90% of explained variance ratio.
4. Modeling</br>
We choose XGboost, random forest, XGBRF, and KNN as our models. Also, linear and logistic regressions are considered as the model baselines.
5. Optimization: Fine-tuning hyperparameters of the selected models by try-and-error. For ensemble learning, we set numbers of tree to 100. For XGBRF and XGBoost, eta is inputed as 0.05 and the objective XGBoosting function is "binary:logistic".
```python=
XGBRFClassifier(n_estimators=100, subsample=0.9, eta=0.05)
XGBClassifier(n_estimators=100, objective='binary:logistic', eta=0.05)
RandomForestClassifier(n_estimators=100)
```
6. 10-fold cross validation

## Experiment Results

