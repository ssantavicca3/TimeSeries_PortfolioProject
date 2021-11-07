## For data
import pandas as pd
import numpy as np
## For plotting
import matplotlib.pyplot as plt
## For outliers detection
from sklearn import preprocessing, svm
## For stationarity test and decomposition
import statsmodels.tsa.api as smt
import statsmodels.api as sm

dtf = pd.read_csv('Data\sales_train.csv')
#print(dtf.head())

# format datetime column
dtf["date"] = pd.to_datetime(dtf['date'], format='%d.%m.%Y')
# create time series
ts = dtf.groupby("date")["item_cnt_day"].sum().rename("sales")
#print(ts.head())
#print(ts.tail())

#ts.plot()


## Trend Analysis

# Plot ts with rolling mean and 95% confidence interval with rolling std.
# :parameter
#   :param ts: pandas Series
#   :param window: num - for rolling stats
#   :param plot_ma: bool - whether plot moving average
#   :param plot_intervals: bool - whether plot upper and lower bounds

def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30,
            figsize=(15,5)):
   rolling_mean = ts.rolling(window=window).mean()
   rolling_std = ts.rolling(window=window).std()
   plt.figure(figsize=figsize)
   plt.title(ts.name)
   plt.plot(ts[window:], label='Actual values', color="black")
   if plot_ma:
      plt.plot(rolling_mean, 'g', label='MA'+str(window),
               color="red")
   if plot_intervals:
      lower_bound = rolling_mean - (1.96 * rolling_std)
      upper_bound = rolling_mean + (1.96 * rolling_std)
   plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound,
                    color='lightskyblue', alpha=0.4)
   plt.legend(loc='best')
   plt.grid(True)
   plt.show()


plot_ts(ts, window=30)
plot_ts(ts, window=365)


## Outliers detection
# Plot histogram
ts.hist(color="black", bins=100)
# Boxplot
ts.plot.box()

#SVM for outliers
# Find outliers using sklearn unsupervised support vetcor machine.
# :parameter
#     :param ts: pandas Series
#     :param perc: float - percentage of outliers to look for
# :return
#     dtf with raw ts, outlier 1/0 (yes/no), numeric index

def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda
                                              x: 1 if x==-1 else 0)
    print(dtf_outliers[dtf_outliers["outlier"]==1].count())  #delete this line later
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found"
           +str(sum(dtf_outliers["outlier"]==1)))
    ax.plot(dtf_outliers["index"], dtf_outliers["ts"],
            color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1]["index"],
               y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'],
               color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers


# Interpolate outliers in a ts.
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5,
         title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black",
                  label="interpolated", legend=True)
    plt.show()
    return ts_clean

## detect the outliers
dtf_outliers = find_outliers(ts, perc = 0.05)
## outliers index position
outliers_index_pos = dtf_outliers[dtf_outliers["outlier"]==1].index
## exclude outliers
ts_clean = remove_outliers(ts, outliers_idx=outliers_index_pos)


# Test stationarity by:
#     - running Augmented Dickey-Fuller test wiht 95%
#     - plotting mean and variance of a sample from data
#     - plottig autocorrelation and partial autocorrelation

def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(15,10)):
    with plt.style.context(style='bmh'):
        ## set figure
        fig = plt.figure(figsize=figsize)
        ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
        acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))

        ## plot ts with mean/std of a sample from the first x%
        dtf_ts = ts.to_frame(name="ts")
        sample_size = int(len(ts)*sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red",
                            linestyle="--", linewidth=0.7)
        ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'],
                y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(ax=ts_ax,
                legend=False, color="red", linewidth=0.9)
        ts_ax.fill_between(x=dtf_ts.head(sample_size).index,
                           y1=dtf_ts['lower'].head(sample_size),
                           y2=dtf_ts['upper'].head(sample_size),
                           color='lightskyblue')

        ## test stationarity (Augmented Dickey-Fuller)
        adfuller_test = sm.tsa.stattools.adfuller(ts, maxlag=maxlag,
                                                  autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+'(p value: '+str(p)+')')

        ## pacf (for AR) e acf (for MA)
        smt.graphics.plot_pacf(ts, lags=maxlag, ax=pacf_ax,
                 title="Partial Autocorrelation (for AR component)")
        smt.graphics.plot_acf(ts, lags=maxlag, ax=acf_ax,
                 title="Autocorrelation (for MA component)")
        plt.tight_layout()

test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30)


# Difference the time series
diff_ts = ts - ts.shift(1)
print(ts.head())
print(diff_ts.head())
diff_ts = diff_ts[(pd.notnull(diff_ts))]
print(diff_ts.head())
test_stationarity_acf_pacf(diff_ts, sample=0.20, maxlag=30)


## Seasonality Analysis
decomposition = smt.seasonal_decompose(ts, freq=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
ax[0].plot(ts)
ax[0].set_title('Original')
ax[0].grid(True)
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[1].grid(True)
ax[2].plot(seasonal)
ax[2].set_title('Seasonality')
ax[2].grid(True)
ax[3].plot(residual)
ax[3].set_title('Residuals')
ax[3].grid(True)
