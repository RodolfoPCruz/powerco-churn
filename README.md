# PowerCo Churn Prediction

Predicting customer churn for PowerCo, a major European energy utility, with a focus on the SME (Small & Medium Enterprise) segment.  The project explores how pricing impacts churn rates and builds a predictive model to help target retention strategies.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Approach & Methodology](#approach--methodology)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Future Work](#future-work)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## 1. Project Overview
Following the liberalization of Europe’s energy market, PowerCo experienced high churn rates among SME customers.  
The objective of this project is to:
- **Predict** which SME customers are at risk of churning.
- **Evaluate** the effect of price changes on churn probability.
- **Support** retention strategies, including a proposed 20% discount for high-risk customers.

By anticipating churn, PowerCo can take proactive measures to retain customers and optimize pricing strategie

## 2. Data Description

> The dataset contains historical information on PowerCo’s SME customers and is provided in two CSV files:

> 1. **clients_data.csv** – contains customer information.
> 2. **price_data.csv** – contains historical energy price data.
>
> ### **Features in `clients_data.csv`**
>
> - **id** – client company identifier
> - **channel_sales** – sales channel code
> - **cons_12m** – electricity consumption over the past 12 months
> - **cons_gas_12m** – gas consumption over the past 12 months
> - **cons_last_month** – electricity consumption in the last month
> - **date_activ** – contract activation date
> - **date_end** – registered contract end date
> - **date_modif_prod** – date of the last product modification
> - **date_renewal** – date of the next contract renewal
> - **forecast_cons_12m** – forecasted electricity consumption for the next 12 months
> - **forecast_cons_year** – forecasted electricity consumption for the next calendar year
> - **forecast_discount_energy** – forecasted value of the current discount
> - **forecast_meter_rent_12m** – forecasted meter rental cost for the next 12 months
> - **forecast_price_energy_off_peak** – forecasted electricity price for the 1st period (off-peak)
> - **forecast_price_energy_peak** – forecasted electricity price for the 2nd period (peak)
> - **forecast_price_pow_off_peak** – forecasted power price for the 1st period (off-peak)
> - **has_gas** – indicates whether the client also has a gas subscription
> - **imp_cons** – current paid consumption
> - **margin_gross_pow_ele** – gross margin on power subscription
>   - Represents the profit margin from selling electricity subscriptions before accounting for overhead costs (e.g., salaries, rent, administration).
>   - Formula: **Gross Margin** = Revenue from power subscription – Direct costs of supplying power
> - **margin_net_pow_ele** – net margin on power subscription
>   - Indicates actual profit after subtracting all expenses (direct and indirect).
>   - Formula: **Net Profit** = Total Revenue – All Costs (direct costs + operating expenses + taxes + interest + depreciation, etc.)
> - **nb_prod_act** – number of active products and services
> - **net_margin** – total net margin
> - **num_years_antig** – client tenure (in years)
> - **origin_up** – code of the initial electricity campaign the customer subscribed to
> - **pow_max** – contracted maximum power
> - **churn** – indicates whether the client churned within the next 3 months
>
> ### **Features in `price_data.csv`**
>
> - **id** – client company identifier
> - **price_date** – reference date
> - **price_off_peak_var** – variable electricity price for the 1st period (off-peak)
> - **price_peak_var** – variable electricity price for the 2nd period (peak)
> - **price_mid_peak_var** – variable electricity price for the 3rd period (mid-peak)
> - **price_off_peak_fix** – fixed power price for the 1st period (off-peak)
> - **price_peak_fix** – fixed power price for the 2nd period (peak)
> - **price_mid_peak_fix** – fixed power price for the 3rd period (mid-peak)
>
> ------
>
> The dataset is owned by **Preparatório para Entrevistas em Dados (PED)**, a course designed to prepare students for data science job interviews. Due to copyright restrictions, I am unable to share the dataset.
>  For more information, please visit https://renatabiaggi.com/ped/.
>
> ------
>
> ## 3. Approach & Methodology
>
> 