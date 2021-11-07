#Tutorial: https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b

## For data
import pandas as pd
import numpy as np
## For plotting
import matplotlib.pyplot as plt
## For Arima
import pmdarima
import statsmodels.tsa.api as smt
## For Lstm
from tensorflow.keras import models, layers, preprocessing as kprocessing
## For Prophet
#from fbprophet import Prophet


# Read in data
dtf = pd.read_csv('Data\sales_train.csv')
dtf.head()
## format datetime column
dtf["date"] = pd.to_datetime(dtf['date'], format='%d.%m.%Y')
## create time series
ts = dtf.groupby("date")["item_cnt_day"].sum().rename("sales")
ts.head()
ts.tail()


### Partitioning

# Split train/test from any given data point.
# :parameter
#     :param ts: pandas Series
#     :param test: num or str - test size (ex. 0.20) or index position
#                  (ex. "yyyy-mm-dd", 1000)
# :return
#     ts_train, ts_test
def split_train_test(ts, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[
                      ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|",
          ts.index[split], "| test size:", perc, " ---")

    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False,
                               sharey=True, figsize=figsize)
        ts_train.plot(ax=ax[0], grid=True, title="Train",
                      color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test",
                     color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()

    return ts_train, ts_test

# Let’s split the data:
ts_train, ts_test = split_train_test(ts, test="2015-06-01")

## Write function to evaluate the performance of our algorithms.
## These should be flexible enough for later use with any kind of
## of time series data (date-time index, numeric index, etc,...)
## The function to evaluate the models: it's a function that expects a dataframe
## as input with input data (column "ts"), fitted values (column "models"),
## predictions (column "forecast")

# Evaluation metrics for predictions.
# :parameter
#     :param dtf: DataFrame with columns raw values, fitted training
#                  values, predicted test values
# :return
#     dataframe with raw ts and forecast
def utils_evaluate_forecast(dtf, title, plot=True, figsize=(20,13)):
    try:
        ## residuals
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]

        ## kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()
        error_mean = dtf["error"].mean()
        error_std = dtf["error"].std()
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()
        mse = dtf["error"].apply(lambda x: x**2).mean()
        rmse = np.sqrt(mse)  #root mean squared error

        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std

        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))

        return dtf[["ts","model","residuals","conf_int_low","conf_int_up",
                    "forecast","error","pred_int_low","pred_int_up"]]

    except Exception as e:
        print("--- got error ---")
        print(e)

## Use auto_arima to find the right parameter combination
best_model = pmdarima.auto_arima(ts,
                                 seasonal=True, stationary=False,
                                 m=7, information_criterion='aic',
                                 max_order=20,
                                 max_p=10, max_d=3, max_q=10,
                                 max_P=10, max_D=3, max_Q=10,
                                 error_action='ignore')
print("best model --> (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)


# Build & train the model and evaluate the predictions on the test set:

# Fit SARIMAX (Seasonal ARIMA with External Regressors):
#     y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] +
#               b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
# :parameter
#     :param ts_train: pandas timeseries
#     :param ts_test: pandas timeseries
#     :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d:
#                   degree of differencing (to remove trend), q: order
#                   of moving average (MA)
#     :param seasonal_order: tuple - (P,D,Q,s) --> s: number of
#                   observations per seasonal (ex. 7 for weekly
#                   seasonality with daily data, 12 for yearly
#                   seasonality with monthly data)
#     :param exog_train: pandas dataframe or numpy array
#     :param exog_test: pandas dataframe or numpy array
# :return
#     dtf with predictons and the model
def fit_sarimax(ts_train, ts_test, order=(1,0,1),
                seasonal_order=(0,0,0,0), exog_train=None,
                exog_test=None, figsize=(15,10)):
    ## train
    model = smt.SARIMAX(ts_train, order=order,
                        seasonal_order=seasonal_order,
                        exog=exog_train, enforce_stationarity=False,
                        enforce_invertibility=False).fit()
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues

    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train),
                            end=len(ts_train)+len(ts_test)-1,
                            exog=exog_test)

    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model


## Let's fit the model on the train set & forecast the same period of the test set:
dtf, model = fit_sarimax(ts_train, ts_test, order=(1,1,1),
                         seasonal_order=(1,0,1,7))


## NOT BAD: when forecasting, the average error of prediction in 394 unit sales
## (17% of the predicted value).

#################################################################################
### LTSM
