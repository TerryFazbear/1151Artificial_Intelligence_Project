# Bank Failure Prediction
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

### EDA
**Number of financial institutions in years**: Nubmer of banks has a downtrend with year-to-year, which might related to 2008 financial crisis.
<img src="https://i.imgur.com/jaIIaWb.png" width="50%" height="50%">

**Default rate of banks**: Banks are either recidivist of defaulting or never default at all.
<img src="https://i.imgur.com/eN0d1RT.png" width="30%" height="30%">

**Default frequency of banks**:
Most of banks have no default record at all.
<img src="https://i.imgur.com/4sSdZPW.png" width="50%" height="50%">

## Literature Review
### Factors for bank failure prediction
### Resampling
## Methodology
1. Data preprocessing
2. Feature selection
3. Scaling and dimension reduction
4. Modeling
5. Optimization
6. K-fold cross validation

## Experiment Results
