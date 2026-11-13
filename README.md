# Advanced AI/ML Project

## Problem and Goal

### Business Problem

Every business, especially grocery, needs to have a forecast on future sales. This will allow for strategic business planning and inventory management. However, current models such as ARIMA and exponential smoothing only model one product’s history at a time. This leads to potential inaccuracy during changes in weather, upcoming events not accounted for, etc. This then leads to unexpected business hurdles; product running its shelf life, too many labor hours scheduled, or the opposite end with out-of-stock items, and lack in workforce.

To solve this, the project will train one deep learning model across all 9,147 products and store combinations at once, instead of one model per product. The model will learn from each item's sales history along with the calendar events, SNAP (Supplemental Nutrition Assistance Program) benefit days, and selling prices that drive changes in demand. It will then produce a 28-day daily forecast, including a likely high and low range, for every product at every store. These forecasts will be displayed in a dashboard where an inventory manager can plan orders, and a scheduler can plan labor hours against the expected sales.

### SMART Goal

**Situation:** A retailer with 3 Texas stores and 3,049 products needs a 28-day daily sales forecast.

**Behavior:** By EOD December 4th, I will train a deep learning model on the Texas stores of the M5 dataset that scores at least 15% lower WRMSSE (weighted root mean squared scaled error) than a seasonal naïve baseline.

**Impact:** A more accurate forecast will allow for inventory managers and schedulers to improve in-stock and labor hours utilization. This will be measured as the total units mis-forecast during the 28-day test period: units over-forecast plus units under-forecast, summed across all 9,147 product and store combinations. The model will reduce that total by at least 10% compared to the seasonal naïve baseline. Over-forecast units represent product at risk of expiring and labor hours scheduled for sales that do not happen, while under-forecast units represent potential out-of-stocks.

## Dataset and Analytical Method

### Dataset

The M5 dataset contains Walmart’s daily unit sales for 3,049 products across 10 stores in California (4), Texas (3), and Wisconsin (3) from 2011 to 2016. The data is split into three CSV files: a sales file with one row per product and store and one column per day (30,490 rows by 1,941 days), a calendar file that maps each day to its date, holidays, events, and SNAP (Supplemental Nutrition Assistance Program) benefit days (1,969 rows), and a sell prices file with the weekly price of each product at each store (roughly 6.8 million rows). Only the three Texas stores (TX_1, TX_2, TX_3) will be used, which leaves 9,147 rows of daily sales, i.e., 3,049 products at each of the three stores.

Two weather sources will be joined to the sales by date. Observed daily highs, lows, and rainfall will come from NOAA’s GHCN-Daily (Global Historical Climatology Network) records for the Dallas/Fort Worth, Houston Intercontinental, San Antonio, and Austin airports. The 7-day forecasts that were issued at the time will come from the National Weather Service’s archived Point Forecast Matrices for those same airports, which the Iowa Environmental Mesonet stores in its NWS text product archive. Any day without an archived forecast will fall back to the seasonal average for that date.

However, M5 does not disclose which cities its stores are in, so there is no way to match a store to its nearest airport. Thus, the four airports will be combined into one equal-weighted Texas average, i.e., every Texas series will share the same weather signal. Unfortunately, this means the model can only learn how Texas weather as a whole moves demand, not how weather differs from one store to another.

**Disclosure:** I will be using an AI assistant (Claude) to help write the scripts that download, parse, and average the NOAA and NWS weather data. Every value comes directly from the public sources cited below, and I will be spot checking the processed data against the original records before it is used for training.

Source: Howard, A., inversion, Makridakis, S., & vangelis. (2020). *M5 forecasting - accuracy* [Data set]. Kaggle. https://www.kaggle.com/competitions/m5-forecasting-accuracy

Source: Menne, M. J., Durre, I., Korzeniewski, B., McNeill, S., Thomas, K., Yin, X., Anthony, S., Ray, R., Vose, R. S., Gleason, B. E., & Houston, T. G. (2012). *Global Historical Climatology Network - Daily (GHCN-Daily), Version 3* [Data set]. NOAA National Centers for Environmental Information. https://doi.org/10.7289/V5D21VHZ

Source: Iowa Environmental Mesonet. (n.d.). *NWS text product archive* [Data set]. Iowa State University. https://mesonet.agron.iastate.edu/wx/afos/

Source: Anthropic. (2026). *Claude* (Opus 5.5) [Large language model]. https://claude.ai

### Analytical Method

The method used will be DeepAR, a LSTM (long short-term memory) based forecasting model. It will be compared against a seasonal naïve baseline and, if time allows, LightGBM.

DeepAR (Salinas et al., 2020) is an autoregressive recurrent neural network that trains one global model across all 9,147 series instead of a separate model for each. For each day, it reads a product’s recent sales at a store plus known inputs (day of the week, events, SNAP days, selling price, and weather), predicts a probability distribution for the next day’s sales, then feeds a value drawn from that distribution back in to forecast the following day. In other words, the output is a range of likely sales instead of a single number, which gives both a point forecast and a prediction interval for all 28 days.

The weather inputs follow what a store would actually know on the day the forecast is made. For the first 7 days of the 28-day window, the model will use the National Weather Service forecast issued that morning. For days 8 through 28, where no forecast exists, it will use the seasonal average for those dates. During training, each day uses the weather that was observed. To show whether weather is worth including, DeepAR will be trained with and without the weather inputs and both versions will be scored.

This method suits the M5 data for three reasons. First, most items sell only a few units per day with many zero-sales days, which leaves a single-series model like ARIMA very little to learn from. A global model lets those sparse items borrow the weekly, holiday, SNAP, and weather patterns of similar items and stores. Second, sales will be modeled with a negative binomial distribution, which fits daily unit sales because they are whole, non-negative numbers whose variance is usually larger than their average. Finally, the prediction interval gives inventory managers a low and high estimate for safety stock instead of a single number.

The seasonal naïve baseline simply repeats last week’s sales for each weekday and sets the minimum bar. LightGBM, the gradient-boosted tree model behind the winning M5 competition entry, is the stronger comparison, but it is an optional stretch goal: it will only be added if DeepAR is finished on schedule. DeepAR will be tuned with rolling-origin cross-validation, i.e., training on earlier days and validating on the next 28 days across three windows.

Source: Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting, 36*(3), 1181–1191. https://doi.org/10.1016/j.ijforecast.2019.07.001

## Project Management and Characteristics

### Project Schedule

| TASK | TIMELINE | DUE | COST |
|---|---|---|---|
| Data exploration and preprocessing | 11/09/2026 to 11/13/2026 | 11/13/26 | $0 |
| DeepAR training and tuning/Baseline Models | 11/14/2026 to 11/27/2026 | 11/27/26 | $0 |
| Evaluation and dashboard | 11/28/2026 to 12/11/2026 | 12/11/26 | $0 |
| Documentation and submission | 12/12/2026 to 12/18/2026 | 12/18/26 | $0 |

### Functional and Nonfunctional Requirements

**Functional:** Load and prepare the data, train and evaluate the models, then display a forecast for any product and store via a Streamlit dashboard.

Loading and preparing the data means filtering the sales file to the three Texas stores, reshaping it from one column per day to one row per product, store, and day, then merging in the calendar, price, and weather data. Categorical fields such as store, department, and event type will be converted to numeric codes. The days before each item was first sold will be removed, since those zeros only mean the item was not on the shelf yet. For DeepAR, each series will also be scaled by its average sales so high and low volume items train on a similar scale.

Training and evaluating means fitting the seasonal naïve baseline and DeepAR (with and without weather) on the same training days, then scoring each one on the final 28 days (days 1,914 to 1,941), which are held out as the test set. The scores will be WRMSSE, RMSE (root mean squared error), MAE (mean absolute error), and total units mis-forecast. The dashboard will let the user pick any Texas store and product, then show the last 90 days of actual sales next to each model’s 28-day forecast, DeepAR’s 80% prediction interval, and a table comparing the models.

**Nonfunctional:** This is trained on a single GPU, the dashboard loads in under 5 seconds, and the results are replicable.

The GPU limit means a full DeepAR training run must fit in GPU memory and finish in under 4 hours, which will be controlled through the batch size and the number of batches per epoch. The dashboard will load in under 5 seconds because it reads forecasts already saved to Parquet files instead of running the model each time a product is selected. Replicable means fixed random seeds, package versions pinned in a requirements.txt file, and README instructions that let anyone with the repository and the Kaggle data reproduce the reported scores (GPU training is not perfectly deterministic, so DeepAR’s scores may vary slightly between runs). The weather data is downloaded by a script in the repository, so it can be rebuilt the same way. Finally, all code will be committed to the project repository at each milestone.

### Risks and Mitigation

**Memory Limits:** the Texas data has about 17.8 million rows once it is reshaped. My PC has 32 GB of RAM, and only 12 GB of VRAM, which is sufficient. Mitigation (if needed): downcast data types and save to Parquet to keep the memory usage low. At 8 bytes per value under the default 64-bit data types, 17.8 million rows by roughly 30 columns is about 4.3 GB, and each merge temporarily copies the table. To keep that down, sales will be stored as 16-bit integers, prices and weather as 32-bit floats, and text fields such as item and store IDs as categories. Removing the days before each item’s first sale will also cut rows before the merges.

**Training Time:** Training on 9,147 series will take some time. Mitigation: tune on a sample of series first and save checkpoints during full training runs. Every tuning trial must train across three cross-validation windows, so the sample will be 10% of the series, drawn evenly from every store and department, and each run will use early stopping to end training once the validation loss stops improving. Only the best settings found on the sample will be trained on the full data.

**Data leakage:** features that use future information would inflate results. Mitigation: use a time-based split and keep the test period untouched until final evaluation. For example, a 7-day rolling average computed over the full dataset would let a training row see the sales it is supposed to predict. The same applies to weather: the observed weather for the forecast window was not known on the day the forecast was made. Thus, every lag and rolling feature will only use sales from before the forecast start date, the forecast window will only use weather forecasts issued by that morning (and seasonal averages after day 7), each cross-validation window will train on earlier days and validate on later days, and the final 28 days will be scored once, after all tuning is finished.

**Weather data gaps:** some archived forecasts may be missing or fail to parse. Mitigation: any missing day falls back to the seasonal average for that date, and the share of days covered by a real forecast will be reported.

**Model underperforms:** DeepAR may not beat the baseline by the target margin, and weather may not improve it at all. Mitigation: tune the model and report the comparison either way. The weather result is reported as a finding whether it helps or not. If DeepAR falls short of the baseline target, the first settings revisited will be the context length (how many past days it reads) and the network size.

## Design and Development

### Languages and Libraries

The language used is Python 3.14.7 with libraries: pandas, PyTorch, PyTorch Forecasting, LightGBM, DeepAR, and Streamlit.

### Development Environment

The IDE (integrated development environment) is Build #PY-253.30387.127, built on January 27, 2026, with a Python Virtual Environment on a MacBook Pro 14-inch M4 Pro Chip, with 24 GB of RAM for coding and updating. The model training on my Windows PC WSL (Windows Subsystem for Linux) running Ubuntu 24.04.5 LTS, using its NVIDIA RTX 3080i GPU (12 GB VRAM). This will be connected and managed over SSH via a Tailscale (tailscale only necessary for off-prem connection) VPN connection between the MacBook and PC.

### Operating Systems and Platforms

This project will be coded and built on a MacBook Pro 14 inch with an M4 Pro Chip and 24 GB of ram. This device is running MacOS Golden Gate 27.2. The model will be trained and run on a WSL VM running Ubuntu 24.04.5 LTS.