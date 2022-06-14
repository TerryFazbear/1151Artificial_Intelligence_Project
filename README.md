# Bank Failure Prediction

 :speech_balloon:*This document is supplementory for AI final project presentation video. Due to the video time limitation of 15 miniutes, the slides and video cannot cover comprehensive experimental details. Hope this attachement can provide more guidence and explaination of our projects. The complete experiment results can be found in our [Google Colab](https://colab.research.google.com/drive/1UOQLgOUAaM31_giy-JzfFfaf0OKooaNj?authuser=1#scrollTo=YEo_fpHEMnXa)*
 
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
- Filter</br>
Remove noise by dropping features that are not correlative with the target, namely, failure. First is to dropping features whose correlation coefficients with target is lower than 0.25%. Next step is deleting any one of two features whose correlation is larger than 0.85 or smaller than -0.85. For example, the correlation of unemployement rate is strong negative to rate of inflation, therefore, we can remove unemplyement rate since these two features can provide us with similar description.

- Wrapper<br>
The drawback of wrapper is the high computational cost. As a result, in order to save computational time, we randomly select 10,000 samples to compute average scores in different number of features. However, sampling will lead to difference experiment results each time. Using backward sequential feature selector with decision tree to compute accuracy of each combination of features. In the below example, we choose 14 features as our selected subset since it has the highest score, which are rssd_id, htminv_assets, lev,  ciloan_assets, demanddep_assets, creditcard_line_obs, month, Size, loanagri_assets, Ciloan_assets, roa, resid_recol_obs, Tier1_car, housing start(million), and year.</br>
**Figure 5: Average scores over different sets of selected features**
<img width="200" alt="image" src="https://user-images.githubusercontent.com/104308942/173612510-1a9ad774-b966-42fd-85ff-f695aebdc91c.png">
- Wrapper + Filter (main feature selection method)
- Lasso</br>
Take Regularization methods, and use Lasso regression (L1) to remove features whose coefficient are 0.

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
For the following experiments, we will test the performance of our propsed methods.
### Experiment A
The purpose of experiment A is to verify after data resampling, the performance of model prediction will become better.

Raw Data|XGBoost|XGBRF|Random Forest
-|-|-|-
Confusion Matrix|<img width="230" alt="image" src="https://user-images.githubusercontent.com/104308942/173615066-2fa4ab4e-9c9c-41fb-90e6-94cf0b5c0bba.png">|<img width="227" alt="image" src="https://user-images.githubusercontent.com/104308942/173614571-d0624b77-12ea-4af2-98ba-1f06948bf613.png">|<img width="235" alt="image" src="https://user-images.githubusercontent.com/104308942/173614918-6dd3a532-80d6-481f-a3a3-5a1833169449.png">
Report|<img width="258" alt="image" src="https://user-images.githubusercontent.com/104308942/173615126-50362be4-7b48-402a-85c3-77c84569e3f4.png">|<img width="256" alt="image" src="https://user-images.githubusercontent.com/104308942/173614829-712403cb-62f7-4fff-8a83-1190605bb577.png">|<img width="265" alt="image" src="https://user-images.githubusercontent.com/104308942/173614990-6b44c357-ef0b-4b4e-8165-8491d8f02775.png">

Smote-Tomek|XGBoost|XGBRF|Random Forest
-|-|-|-
Confusion Matrix|<img width="229" alt="image" src="https://user-images.githubusercontent.com/104308942/173615548-01d1a668-8247-4f7c-b65e-d950e4bf62b2.png">|<img width="231" alt="image" src="https://user-images.githubusercontent.com/104308942/173615683-d75a0ebf-2ddc-44d8-98ad-f7432ada0d52.png">|<img width="232" alt="image" src="https://user-images.githubusercontent.com/104308942/173615873-acc69fdd-a998-40df-9db5-7e4c38dec413.png">
Report|<img width="258" alt="image" src="https://user-images.githubusercontent.com/104308942/173615591-4f9bb960-51eb-44a5-a9af-b3b5e563fecc.png">|<img width="253" alt="image" src="https://user-images.githubusercontent.com/104308942/173615772-d8f0835a-f5bf-43b9-9ffc-0edbfaaafe84.png">|<img width="255" alt="image" src="https://user-images.githubusercontent.com/104308942/173616043-8bc4b371-014e-4f30-90c0-376c56a46aea.png">

Smote-ENN|XGBoost|XGBRF|Random Forest
-|-|-|-
Confusion Matrix|<img width="228" alt="image" src="https://user-images.githubusercontent.com/104308942/173616472-953b3986-6c36-4b8c-adc5-2cd9f7de446f.png">|<img width="234" alt="image" src="https://user-images.githubusercontent.com/104308942/173616663-ab5985a3-e2f1-4ad3-a719-a74289a83b37.png">|<img width="230" alt="image" src="https://user-images.githubusercontent.com/104308942/173616739-171315de-1ef3-4e75-8aa5-20193a99258f.png">
Report|<img width="257" alt="image" src="https://user-images.githubusercontent.com/104308942/173616547-9db02ec1-2402-4323-a189-769c9b3e237b.png">|<img width="259" alt="image" src="https://user-images.githubusercontent.com/104308942/173616690-fad8d0c8-22f3-427d-bebf-3dfbd92f8e10.png">|<img width="254" alt="image" src="https://user-images.githubusercontent.com/104308942/173616783-2bc517b8-a468-4943-bd20-83bbfd4117a5.png">

### Experiment B
In this part, we will compare several feature selection methods with our proposed method, Wrapper combined Filter, by using XGBoost as the tested model.
We can oberseve that after combining filter with wrapper, the result becomes better, from macro avg 0.77 to 0.86.

Raw|Filter|Wrapper|Wrapper+Filter|Lasso
-|-|-|-|-
<img width="267" alt="image" src="https://user-images.githubusercontent.com/104308942/173617501-f831ad57-a2ef-487d-b8dd-eb806327bce9.png">|<img width="264" alt="image" src="https://user-images.githubusercontent.com/104308942/173617552-bc01ac77-2c39-42b8-9d55-a89dcab7ecb4.png">|<img width="264" alt="image" src="https://user-images.githubusercontent.com/104308942/173617617-fa491a7b-e030-458c-a8fc-043ed7369eec.png">|<img width="255" alt="image" src="https://user-images.githubusercontent.com/104308942/173617703-45b7bf41-564e-43bd-89e2-3bb72b6ff843.png">|<img width="259" alt="image" src="https://user-images.githubusercontent.com/104308942/173617756-72f36359-6087-427b-be9e-87b6a7c45887.png">

### Experiment C
We want to compare PCA with feature selection by taking XGBoost as the experimental model. Since PCA incorporates noise and create new dimension that are not real, creating an erroneous result, but feature selection removes noises by dropping variables with low relation with the target and better to perform model explainability. Therefore, we maintain that feature selection will give a better performance compared to PCA. 

PCA|Feature selection|PCA+Feature Selection
-|-|-
<img width="257" alt="image" src="https://user-images.githubusercontent.com/104308942/173619602-8e9865e6-c320-4e3e-95ef-58148c7f7995.png">|<img width="257" alt="image" src="https://user-images.githubusercontent.com/104308942/173619876-f1a9ade6-75d3-4eba-be84-38f4c1ed3c97.png">|<img width="254" alt="image" src="https://user-images.githubusercontent.com/104308942/173619999-a89ccf54-5a5e-422b-b69a-b2d051e0dabc.png">

The experiment result verify our hypothesis. Features selection has 0.88 macro average, which outperforms 0.81 in PCA.

### Experiment D
All the models outperform the baselines and Random Forest are the best among all with 0.94 accuracy and macro average scores.

XGBRF|Random Forest|XGBoost|KNN|Linear Regression|Logistic Regression
-|-|-|-|-|-
<img width="253" alt="image" src="https://user-images.githubusercontent.com/104308942/173621580-6940eb2f-e311-4442-854b-de563ed49443.png">|<img width="260" alt="image" src="https://user-images.githubusercontent.com/104308942/173621649-354426d0-6bf2-4d5f-acb5-852176fad395.png">|<img width="251" alt="image" src="https://user-images.githubusercontent.com/104308942/173621716-1a4a1b67-26ed-4980-b87a-4e672e9e4901.png">|<img width="255" alt="image" src="https://user-images.githubusercontent.com/104308942/173621794-3f7a0ded-877e-4631-b121-064bd824e752.png">|<img width="256" alt="image" src="https://user-images.githubusercontent.com/104308942/173621943-392618a8-bcf2-4679-a287-8c3b2b80723b.png">|<img width="254" alt="image" src="https://user-images.githubusercontent.com/104308942/173621881-3fc2a89d-2c59-4c3a-9539-465950117175.png">
