# Primary tutorial: https://towardsdatascience.com/arima-vs-lstm-forecasting-electricity-consumption-3215b086da77

#####################################
#                                   #
#     Get Started: Skip to...       #
# PRIMARY/UPDATED FUNCTIONS SECTION #
#                                   #
#####################################

###########################################################################
#                                                                         #
# The following needs to be run before some of the functions (e.g.,       #
# eval_forecast, plot_eval_forecast). NB: sub correct info in.            #
#                                                                         #
# train_test_split(ms_ts, split_perc = 0.85)                              #
# tbats_model <- tbats(train)                                             #
# forecast <- forecast(tbats_model, h=length(test), level = c(80, 95, 99))#
#                                                                         #      
###########################################################################


# For sample subsetting & forecasting
library(forecast)   
library(keras)        #LSTM
library(tensorflow)   #LSTM

# For visualizations
library(ggplot2)
library(RColorBrewer)
library(patchwork)     #multi-panel ggplots
library(ggpmisc)       #ggplot tibbles

# For gen. purposes
library(dplyr)         #manipulation
library(magrittr)      #pipe; subsetting secondary series w/in ggplot
library(wrapr)         #pipe ggplot layers
library(glue)          #string magic
library(stringr)       #string magic
library(zoo)           #time-series object w/ 'daily' datetime index
library(data.table)    #%like%


## NOTE:
# may also want TBATS model for multiple seasonalities (weekly, and annual)
# https://stats.stackexchange.com/questions/144158/daily-time-series-analysis

## NOTE:
# after completing everything, rearrange/organize the program, potentially
# with the functions at the top and the testing below it. Plus, can consolidate
# certain code by just extending the functions I've written to use them. E.g.,
# the acf_pacf plot function below can be used instead of manually plotting it
# up here near the top of the program.
## NOTE:
# at the end, if I have a really good set of functions, consider including 
# stop/warning calls to the console within the function(s). This will also help
# me see what I'm missing when I get errors trying to run the functions.
## NOTE:
# to automate the axis-scaling in my functions (for different ts's) just create
# a parameter in the function definition and set it to NULL. That way, I can just
# set them after viewing earlier plots bc I'll have an idea of what the x,ylims
# should be for that ts.


# Import data
df <- read.csv("Data/sales_train.csv")
# Format datetime column
df$date <- as.Date(df$date, "%d.%m.%Y")        #for ordering


## Create "sales" series with a daily frequency datetime index
# Create time series by aggregating daily sales
ts_df <- df %>%
  group_by(date) %>% 
  summarise(sales = sum(item_cnt_day))

## Create time-series objects 
#zoo: no forced seasonality, good w/ daily data
z_ts <- read.zoo(ts_df, format="%Y-%m-%d")     
#ts; forced seasonality, crap with daily data (needed for 's' in auto.arima())
ts <- ts(ts_df$sales, start=as.Date("2013-01-01"), frequency = 7)  
#msts; account for multiple seasonal patterns (needed for TBATS)
ms_ts <- msts(ts_df$sales, seasonal.periods = c(7,365.25))           


## Predict sales and plot forecast 
# Write a function to split ts and output train and test sets (NOTE: COULD JUST MERGE THESE TWO F'Ns)
# option A: output a single series(test/train) with a call. Could be used in-line
train_test_split <- function (ts, split_perc=0.85, out.train=F, 
                              out.test=F, full_df=F, ts_col=NULL) {
  
  # input either ts object or ts df
  if (full_df) {
    train <- head(ts, round(length(ts_col) * split_perc))      
    h <- length(ts_col) - round(length(ts_col) * split_perc)
    test <- tail(ts, h) 
  } else {
    train <- head(ts, round(length(ts) * split_perc))      
    h <- length(ts) - length(train)
    test <- tail(ts, h)
  }
  # return train or test
  if (out.train) {
    return(train)
  } 
  if (out.test) {
    return(test)
  }
  
}
train <- train_test_split(ts, split_perc = 0.85, out.train=T)
test <- train_test_split(ts, split_perc = 0.85, out.test=T)
# option B: output both train + test to the global environment with a single call.
train_test_split <- function (ts, split_perc=0.85, full_df=F, ts_col=NULL) {
   
  # input either ts object or ts df
  if (full_df) {
    train <- head(ts, round(length(ts_col) * split_perc))      
    h <- length(ts_col) - round(length(ts_col) * split_perc)
    test <- tail(ts, h) 
  } else {
    train <- head(ts, round(length(ts) * split_perc))      
    h <- length(ts) - length(train)
    test <- tail(ts, h)
  }
  # assign both train and test to global env
  assign("train", train, envir = .GlobalEnv)
  assign("test", test, envir = .GlobalEnv)
  
}  

# Plot training and test set
split_perc <- round(length(train)/(length(train)+length(test)), 2) #variable (subject to value ussed in train_test_split())
par(mfrow=c(2,1))
plot(train, 
     main=glue("Training Set: {100*split_perc}%"), xlab="", ylab="", sub=glue("n={length(train)}"),
     ylim=c(min(min(train), min(test)),
            max(max(train), max(test))))
plot(test, 
     main=glue("Testing Set: {100*(1-split_perc)}%"), xlab="", ylab="", sub=glue("n={length(test)}"),
     ylim=c(min(min(train), min(test)),
            max(max(train), max(test))))

# Plot ACF for training set
ggAcf(train, lag.max=30, ci=0.95) +
  labs(y = "", 
       title = "Autocorrelation",
       subtitle = "CI: 95%") +
  theme_minimal() +
  theme(plot.title = element_text(hjust=0.5, size=12, face="bold"),
        plot.subtitle = element_text(hjust=0.5, size=11, face="italic"))
#NB: it looks like m=7 would be the seasonal comp. of our ARIMA model.

# Use stepwise to select the ARIMA model w/ lowest AIC
fit_arima <- auto.arima(train, max.order = 20, 
                        max.p = 10, max.d = 3, max.q = 10,
                        max.P = 10, max.D = 3, max.Q = 10,
                        stepwise = TRUE, seasonal = TRUE, stationary = FALSE,
                        ic = "aic", trace = TRUE) #try w/o stepwise as well (takes >30mins)

# Predict next X days of sales 
forecast <- forecast(fit_arima, h = length(test), level = c(80, 95, 99))

## Plot predictions with ggplot
# Define dataframe with training (x), forecast (y) and interval (pi) data
len.x <- length(forecast$x)
len.y <- length(forecast$mean)

fc_df <- tibble(date = ts_df$date,
             x = c(forecast$x, rep(NA, len.y)),
             fitted = c(forecast$fitted, rep(NA, len.y)),
             forecasted = c(rep(NA, len.x), forecast$mean),
             lo.80 = c(rep(NA, len.x), forecast$lower[, 1]),
             up.80 = c(rep(NA, len.x), forecast$upper[, 1]),
             lo.95 = c(rep(NA, len.x), forecast$lower[, 2]),
             up.95 = c(rep(NA, len.x), forecast$upper[, 2]),
             lo.99 = c(rep(NA, len.x), forecast$lower[, 3]),
             up.99 = c(rep(NA, len.x), forecast$upper[, 3]))

# Plot training, fitted and forecast data
model = glue("{stringr::str_extract(forecast$method, '[A-Z]+' )}")
date.breaks = "6 months"
date.format = "%b-%Y"
y.breaks = c(0, 2500, 5000, 7500, 10000, 12500)
line.cols = c("black", "darkcyan", "goldenrod1")
forecast %.>% { .$level } %.>%  length(.) %.>% 
  { shade.cols <- brewer.pal(., "PuBuGn") }
main.title = glue("{model} Forecast of Daily Sales")
sub.title = glue("Forecast horizon: {length(test)} days \n
                 Specification: {forecast$method}")
caption = "Data source: Kaggle.com"
x.title = "Date"
y.title = "Sales"
show.points = FALSE
if (show.points) {
  points_set <- geom_point(data = fc_df, aes(date, forecasted, colour = "Forecast"), size = 1)
  points_size <- geom_point(size = 1)
} else {
  points_set <- NULL
  points_size <- NULL
}

ggplot(fc_df,  aes(date, x)) +
  geom_line(aes(colour = "Training")) +
  geom_line(data = fc_df, aes(date, fitted, colour = "Fitted"), size = 0.75) +
  geom_ribbon(data = fc_df, aes(date, ymin = lo.99, ymax = up.99, fill = "99%")) +
  geom_ribbon(data = fc_df, aes(date, ymin = lo.95, ymax = up.95, fill = "95%")) +
  geom_ribbon(data = fc_df, aes(date, ymin = lo.80, ymax = up.80, fill = "80%")) +
  geom_line(data = fc_df, aes(date, forecasted, colour = "Forecast"), size = 0.75) +
  points_set + 
  points_size + 
  scale_y_continuous(breaks = y.breaks) +
  scale_x_date(breaks = seq(fc_df$date[1], fc_df$date[length(fc_df$date)],
                            by = date.breaks),
               date_labels = date.format) +
  scale_colour_manual(name = "Model Data",
                      values = c("Training" = line.cols[1],
                                 "Fitted" = line.cols[4],
                                 "Forecast" = line.cols[3]),
                      breaks = c("Training", "Fitted", "Forecast")) +
  scale_fill_manual(name = "Forecast Intervals",
                    values = c("99%" = shade.cols[1], 
                               "95%" = shade.cols[2],
                               "80%" = shade.cols[3])) +
  guides(colour = guide_legend(order = 1), fill = guide_legend(order = 2)) +
  labs(title = main.title,
       subtitle = sub.title,
       caption = caption,
       x = x.title,
       y = y.title) +
  theme.fxdat +
  theme(plot.subtitle = element_text(lineheight = 0.55))   #closing gap b/w subtitles


## Compare predictions to test set
# Write function comparing the rmspe to the test mean
fc_accuracy_print <- function (test, forecast) {
  
  # calculate the root mean squared prediction error
  rmspe <- test %>%
    {(coredata(.) - forecast$mean)^2} %>%
    mean(.) %>%
    sqrt(.)
  # calculate the mean of the test set
  test_mean <- test %>% coredata(.) %>% mean(.)
  # calculate the test mean percentage account for by the rmspe
  acc_perc <- round(100*(rmspe/test_mean), 1)
  
  # print results to console
  str1 <- glue("With the mean daily sales across the test set of {round(test_mean,1)},
  our RMSPE of {round(rmspe, 1)} accounts for {acc_perc}% of the 
  test mean, which indicates that the performance of the ARIMA 
  model here")
    
  if (acc_perc < 5) {
    str2 <- "is out of this world."
  } else if (acc_perc < 15) {
    str2 <- "is pretty darn good."
  } else if (acc_perc < 40) {
    str2 <- "could be much better."
  } else {
    str2 <- "is basically utter garbage."
  }
  
  print("Accuracy Report:")
  cat(paste(str1, str2))
  
  # Plot test set vs predictions
  acc_df <- tibble(days = 1:length(test),
                   test = coredata(test),
                   predictions = forecast$mean)
  # set y.axis upper bound for custom breaks
  if (between(max(forecast$mean), 500, 1000)) {
    y.breaks <- scale_y_continuous(breaks = seq(0, 1000, by=(1000)/5))
  } else if (between(max(forecast$mean), 1000, 5000)) {
    y.breaks <- scale_y_continuous(breaks = seq(0, 5000, by=(5000)/5))
  } else if (between(max(forecast$mean), 5000, 10000)) {
    y.breaks <- scale_y_continuous(breaks = seq(0, 10000, by=(10000)/5))
  } else if (max(forecast$mean) > 10000) {
    y.breaks <- scale_y_continuous(breaks = seq(0, (max(forecast$mean) + .10*max(forecast$mean)),
                                                by=2500))
  }
  # set some custom subtitle
  sub.title = glue("Forecast horizon: {length(test)} days \n
                 Specification: {forecast$method}")
  
  ggplot(acc_df, aes(x = days)) +
    geom_line(aes(y = test, color = "Test set"), 
              size = 1) +
    geom_line(aes(y = predictions, color = "Predictions"), 
              size = 1) +
    # scale_y_continuous(breaks = c(0, 2500, 5000, 7500, 10000, 12500)) +
    scale_colour_manual(name = "Model Data",
                        values = c("Test set" = "darkblue",
                                   "Predictions" = "darkorange"),
                        breaks = c("Test set", "Predictions")) +
    labs(title = "Test vs. Prediction",
         subtitle = sub.title,
         caption = glue("Predictions based on training set of n={length(train)}"),
         x = "Days",
         y = "Sales") +
    theme.fxdat +
    theme(plot.subtitle = element_text(lineheight = 0.55))   #closing gap b/w subtitles
  
}

fc_accuracy_print(test, forecast)

##-----------------------------------------------------------------------------
## Notes:
##
## Results from trying different combinations of the 3 ts-like objects with 
## seasonalities differing, and the two diff models (ARIMA, TBATS).
#
## USING 60:40 SPLIT: 
#yea 53.2% is pretty damn bad with the zoo object.
#52% isn't much better with the ts object (weekly seasonality).
#using the msts object with multiple seasonalities specified, 
#   auto.arima chose annual seasonality (can choose just 1 or neither;
#   effectively the same as a ts object with frequency=365),
#   for a 52% accuracy.
#using tbats model with both seasonalities still results in very smooth 
#   prediction line converged on the mean. Accuracy is 53.4%
#
## USING 80:20 SPLIT:
#tbats model with 80:20 split gets us down to 45.6% accuracy, which isn't
#   much but the actual forecast looks much better. The prediction line
#   and the PI cones are decent sizes and not only represent seasonality, 
#   but also trend up and down with the test series.
#ARIMA model with 80:20 using the zoo object with no seasonality gets us to 
#   35.1% accuracy. The prediction line looks smooth on the mean again, with
#   massive PI cone.
#ARIMA model with 80:20 using the ts object and weekly seasonality gets us
#   38.4% accuracy. The prediction line looks better, representing weekly 
#   seasonality, however the PI cone is even more massive than before.
#
#MOVING ON TO OTHER APPROACHES TO IMPROVE ACCURACY...
#
##-----------------------------------------------------------------------------

## Try TBATS model instead...
# 'tbats' explicitly models multiple types of seasonality.
#https://stats.stackexchange.com/questions/144158/daily-time-series-analysis
train <- train_test_split(ms_ts, split_perc = 0.85, out.train = T)
test <- train_test_split(ms_ts, split_perc = 0.85, out.test = T)
tbats_model <- tbats(train, use.trend = NULL)
forecast <- forecast(tbats_model, h=length(test), level = c(80, 95, 99))
#plot(forecast)
      # playing around below (copy & pasted this up here on 2.10.22)
eval_forecast(ts, forecast, test=test, train=train, console=T, return.eval_tbl=F, print.eval_tbl=F)
fc_accuracy_print(test, forecast)
plot_eval_forecast(ts, forecast, test, train, og_df.date_col = ts_df$date) 

############################################################################################################
## Try manually configuring the ARIMA model (manually select p,d,q)

# As we did with the full sample in Step 1, look at the PACF, and ACF(for diff'd series)
#the diff'd series removes the trend and seasonality components of the series.
plot_acf_pacf <- function (ts_df, ci=0.95, diff=T, diff_lvl=1) {
  
  ts <- ts_df
  
  # create differenced series 
  diff_ts <- diff(ts_df$sales, differences = diff_lvl)
  diff_ts <- data.frame(diff_ts, ts_df$date[(diff_lvl+1):length(ts_df$date)])
  colnames(diff_ts) <- c("sales", "date")
  
  if (diff) {
    ts_acf <- diff_ts
    labs_acf <- labs(y="", title="Autocorrelation (for MA component)",
                     subtitle=glue("Differenced series - order: {diff_lvl}"))
  } else {
    ts_acf <- ts_df
    labs_acf <- labs(y="", title="Autocorrelation (for MA component)")
  }
  
  # plot acf&pacf
  lower_ylim <- min(min(acf(ts_acf$sales, plot=F)$acf),
                    min(pacf(ts_acf$sales, plot=F)$acf))
  
  acf_plot <- ggAcf(ts_acf$sales, lag.max=30, ci=ci) + 
    ylim(c(lower_ylim, 1)) +
    geom_point(aes(x=lag, y=Freq), color="blue", shape=21, size=1.5, stroke=1.5, fill="white") +
    labs_acf +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold"),
          plot.subtitle = element_text(hjust=0.5, size=10)) 
  pacf_plot <- ggPacf(ts$sales, lag.max=30, ci=ci) + 
    ylim(c(lower_ylim, 1)) +
    geom_point(aes(x=lag, y=Freq), color="blue", shape=21, size=1.5, stroke=1.5, fill="white") +
    labs(y="", title="Partial Autocorrelation (for AR component)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold"))
  
  (acf_plot + pacf_plot) #patchwork::
}

plot_acf_pacf(ts_df, ci=0.95, diff=T)
##-----------------------------------------------------------------------------
# p. in the PACF we see a sudden drop at lag 2, so p=2.
# d. the ACF showed evidence of stationarity by first-differencing the series.
#    d=1.
# q. there isn't a significant positive correlation for the first few lags.
#    The correlation doesn't 'drop off' as it did in the PACF. Therefore, q=0
#    (there is no moving average term).
#
# We will assume that the seasonal P,D,Q parameters are the same as those 
#   identified above, and it would be nice to get the seasonal factor of m=7.
##-----------------------------------------------------------------------------

## Manually fit the ARIMA model
train <- train_test_split(ts, split_perc = 0.85, out.train = T)
test <- train_test_split(ts, split_perc = 0.85, out.test = T)
fit_arima <- arima(train,
                   order=c(2,1,0),
                   seasonal=list(order=c(2,1,0), period=7))  #pretty much trash
fit_arima <- arima(train, 
                   order=c(2,1,1), 
                   seasonal=list(order=c(1,1,1), period=7))  #abt as good as I can get it w/ zoo obj
fit_arima <- arima(train, 
                   order=c(1,1,1), 
                   seasonal=list(order=c(1,0,1), period=7))  #trying to improve w/ testing

# Check to see how tight the lags are within the residual CIs (ACF/PACF)
tsdisplay(residuals(fit_arima), lag.max=30, main='Seasonal Model Residuals')

# Forecast & plot
forecast <- forecast(fit_arima, h = length(test), level = c(80, 95, 99))
# Execute testing functions (eval_forecast() and plot_eval_forecast() are defined below)
#eval_tbl_arima <- eval_forecast(ts, forecast, test=test, train=train, console=F, return.eval_tbl=T, print.eval_tbl=F)
#eval_tbl_arima
eval_forecast(ts, forecast, test=test, train=train, console=T, return.eval_tbl=F, print.eval_tbl=F)
fc_accuracy_print(test, forecast)
plot_eval_forecast(ts, forecast, test, train, og_df.date_col = ts_df$date) 

##-----------------------------------------------------------------------------
## Notes:
#
## 80:20 split:
#  - (1,1,0)(1,1,0)[7] got to 39.4% accuracy.
#  - (3,1,3)(1,1,1)[7] finally got all lags in the residual CIs, but a 55.3% accuracy
#  - (3,1,3)(1,2,1)[7] doesn't have all lags in the res CIs (close though), and has
#    41.8% accuracy.
#  - (3,1,3)(1,2,1)[7] looks decent in the residual CIs, with 39.6% accuracy.
## 85:15 split:
#  - (2,0,0)(4,1,1)[7] looks decent in the residual CIs, with 31.3% accuracy.
##-----------------------------------------------------------------------------

# With some comparable forecast plots, could lay them out in a multi-panel view to compare,
#   with relevant accuracy statistics labeled on the plots somewhere as well.


##########################################################################################################

#####################################
#                                   #
# PRIMARY/UPDATED FUNCTIONS SECTION #
#                                   #
#####################################

## Trying out the follow-up tutorial from the Python guy I started with
# https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b

#NB: have to run everything up to the "Plot the Training & Test Set" section


### For (S)ARIMA(X) stuff...
plot_eval_forecast <- function(ts, forecast, test=test, train=train, og_df.date_col=NULL) {
  
  # construct tmp df
  eval_df <- tibble(date = og_df.date_col,
                    raw = ts,
                    model = c(forecast$fitted, rep(NA, length(test))),  #NEED TO DROP/ADD THE +1 - used it for lstm
                    forecasted = c(rep(NA, length(train)), forecast$mean))
  # residuals
  eval_df$residuals <- eval_df$raw - eval_df$model  #i.e., forecast$residuals
  #eval_df$residuals <- forecast$residuals
  eval_df$error <- eval_df$raw - eval_df$forecasted
  eval_df$error_pct <- eval_df$error / eval_df$raw
  
  # kpis
  residuals_mean <- eval_df %.>% .$residuals %.>% mean(., na.rm = T)
  residuals_sd <- eval_df %.>% .$residuals %.>% sd(., na.rm = T)
  error_mean <- eval_df %.>% .$error %.>% mean(., na.rm = T)
  error_sd <- eval_df %.>% .$error %.>% sd(., na.rm = T)
  
  mae <- eval_df %.>% .$error %.>% abs(.) %.>% mean(., na.rm = T)
  mape <- eval_df %.>% .$error_pct %.>% abs(.) %.>% mean(., na.rm = T)
  mse <- eval_df %.>% .$error %.>% .^2 %.>% mean(., na.rm = T)
  rmse <- mse %.>% sqrt(.)
  
  # intervals
  eval_df$ci_lo <- eval_df$forecasted - 1.96*residuals_sd #confidence
  eval_df$ci_up <- eval_df$forecasted + 1.96*residuals_sd 
  eval_df$pi_lo <- eval_df$forecasted - 1.96*error_sd #prediction
  eval_df$pi_up <- eval_df$forecasted + 1.96*error_sd
  
  # plot results
  theme_4panel <- function(base_size = 12,
                           base_family = ""){
    theme_minimal(base_size = base_size,
                  base_family = base_family) %+replace%
      theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=2, size=11),
        panel.border = element_rect(color="black", fill=NA, size=.5),
        legend.position = NULL,
          legend.key.size = unit(1, "lines"),
          legend.key.height = unit(0.7, "lines"),
          legend.key.width = unit(0.7, "lines"),
          legend.margin = margin(0, 0.1, 0.05, 0.1, "cm"),
          legend.background = element_rect(color = 'black', 
                                           fill = 'white',
                                           linetype = 'solid'),
          legend.title = element_blank(),
          legend.text = element_text(size = 10)
      )
  } #customizing theme_minimal()
               
  #...training
  p1 <- ggplot(eval_df[1:length(train),], aes(x=date)) +
    geom_line(aes(y=raw, color="ts"), size=1) +
    geom_line(aes(y=model, color="model"), size=1) +
    labs(x=NULL, y='', title='Model') +
    scale_color_manual(name="",
                       values=c("ts"="black",
                                "model"="#3399FF")) +
    #might have to automate ylim again
    ylim(c(-500, 15000))  +
    theme_4panel() +
    theme(legend.position = c(.075,.89)) +
    scale_x_date(breaks="6 months", date_labels = "%b-%Y",
                 limits=c(eval_df$date[1], max=eval_df$date[length(train)]),
                 expand=c(0,0)) 
    
  #...test
  p2 <- ggplot(eval_df[length(train):length(eval_df$raw),], aes(x=date)) +
    geom_ribbon(aes(ymin=pi_lo, ymax=pi_up), fill="blue", alpha=0.3) +
    geom_ribbon(aes(ymin=ci_lo, ymax=ci_up), fill="lightblue", alpha=0.6) +
    geom_line(aes(y=raw, color="ts"), size=1) +
    geom_line(aes(y=forecasted, color="forecast"), size=1) +
    labs(x='', y='', title='Forecast') +
    scale_color_manual(name="",
                       values=c("ts"="black",
                                "forecast"="red")) +
    ylim(c(-500, 15000)) +
    theme_4panel() +
    theme(legend.position = c(.91,.89)) +
    scale_x_date(breaks="1 month", date_labels = "%b",
                 limits=c(eval_df$date[length(train)], max=max(eval_df$date)),
                 expand=c(0,0)) 
  #...residuals
  p3 <- ggplot(eval_df, aes(x=date)) +
    geom_line(data = eval_df[1:length(train),],
              aes(y=residuals, color="residuals"), size=1) +
    geom_line(data = eval_df[length(train):length(eval_df$raw),],
              aes(y=error, color="error"), size=1) +
    labs(x='', y='', title='Residuals') +
    scale_color_manual(name="",
                       values=c("residuals"="#3399FF",
                                "error"="red")) +
    theme_4panel() +
    theme(legend.position = c(.095,.89)) +
    scale_x_date(breaks="6 months", date_labels = "%b-%Y",
                 limits=c(eval_df$date[1], max=max(eval_df$date)),
                 expand=c(0,0)) 
  #...residuals distribution
  p4 <- ggplot(eval_df) +
    geom_density(data = eval_df[length(train):length(eval_df$raw),],
                 aes(x=error, color="error"), alpha=0.1, size=1, fill="red") +
    geom_density(data = eval_df[1:length(train),],
                 aes(x=residuals, color="residuals"), alpha=0.2, size=1, fill="#3399FF") +
    labs(x='', y='', title='Residuals Distribution') +
    geom_vline(aes(xintercept=mean(error, na.rm=T), color="error"), 
               size=1, linetype="dashed") +
    geom_vline(aes(xintercept=mean(residuals, na.rm=T), color="residuals"), 
               size=1, linetype="dashed") +
    scale_color_manual(name="",
                       values=c("residuals"="#3399FF",
                                'error'="red")) +
    scale_fill_manual(name="",
                      values=c('residuals'="#3399FF",
                               'error'="red")) +
    theme_4panel() +
    theme(legend.position = c(.905,.89)) 
  
  
  # create subtitle reporting custom model specification
  str_index <- function (forecast, start=NULL, stop=NULL, int=TRUE) {
    #i=14,16,18 are standard seasonal parameters for forecast$method with ARIMA
    if (int) {
      as.integer(substr(forecast$method, 
                        start=start, 
                        stop=stop))
    } else {
      substr(forecast$method, 
             start=start, 
             stop=stop)
    }
    
  }
  
  if (str_index(forecast, start=1, stop=5, int=F) == "ARIMA") {
    first_s_value <- str_index(forecast, start=14,stop=14, int=F)
    if (first_s_value != "" || !is.na(first_s_value)) {
      #if expanding this for exogenous covariate series, condition "X" as well
      cu.subtitle <- paste("S", sep = "", forecast$method)
    }
  } else {
    cu.subtitle <- forecast$method
  }
  
  # display plot panel
  (p1 + p2) / 
  (p3 + p4) + 
    plot_annotation(
      title = 'Evaluation of Model Performance', 
      subtitle = paste('Model Specification:', sep=" ", cu.subtitle),
      caption = glue('Training set :  n={length(train)} ({round(100*(length(train)/(length(train)+length(test))))}%)
                   Test set :         n={length(test)} ({round(100*(length(test)/(length(train)+length(test))))}%)'),
      theme = theme(plot.title = element_text(hjust=0.5, size=15, face="bold"),
                    plot.subtitle = element_text(hjust=0.5, size=12)))
  
  # Alternative color options:
    # darkish green: #009900
    # good blue: #3399FF        
    # thicker turquiose: #00CCCC
    # deeper near-burghandy red: #CC3333
    # brighter off-red: #FF6666
  
}

#NB: to run this w/o error, manually type test=test & train=train, after using the environment variable version of train_test_split()
# eg)
plot_eval_forecast(ts, forecast, test, train, og_df.date_col = ts_df$date) 



##### Create function to fit and compare multiple models (using eval_forecast())
# First create function to evaluate algorithm performance
eval_forecast <- function (ts, forecast, test=test, train=train, console=TRUE, 
                           assign.eval_tbl=FALSE, eval_tbl.name="eval_tbl",
                           print.eval_tbl=FALSE, return.eval_tbl = FALSE) {
  
  # construct tmp df
  eval_df <- tibble(raw = ts,
                    model = c(forecast$fitted, rep(NA, length(test))),   #NEED TO DROP/ADD THE +1 - used it for lstm
                    forecasted = c(rep(NA, length(train)), forecast$mean)) 
  # residuals
  eval_df$residuals <- eval_df$raw - eval_df$model  #i.e., forecast$residuals
  eval_df$error <- eval_df$raw - eval_df$forecasted
  eval_df$error_pct <- eval_df$error / eval_df$raw
  
  # kpis
  residuals_mean <- eval_df %.>% .$residuals %.>% mean(., na.rm = T)
  residuals_sd <- eval_df %.>% .$residuals %.>% sd(., na.rm = T)
  error_mean <- eval_df %.>% .$error %.>% mean(., na.rm = T)
  error_sd <- eval_df %.>% .$error %.>% sd(., na.rm = T)
  
  mae <- eval_df %.>% .$error %.>% abs(.) %.>% mean(., na.rm = T)
  mape <- eval_df %.>% .$error_pct %.>% abs(.) %.>% mean(., na.rm = T)
  mse <- eval_df %.>% .$error %.>% .^2 %.>% mean(., na.rm = T)
  rmse <- mse %.>% sqrt(.)
  
  # print results to console or to a table
  #print to console
  if (console) {
    cat(glue("
    ------------------------------------
    Evaluation of Algorithm Performance
    ------------------------------------
    Training: 
      residuals mean: {round(residuals_mean)}
      sd: {round(residuals_sd)}\n
    Test: 
      error mean: {round(error_mean)}
      sd: {round(error_sd)}
      mae: {round(mae)}
      mape: {round(mape*100)}%
      mse: {round(mse)}
      rmse: {round(rmse)}
      test ratio:
        test mean: {round(mean(test))}
        rmse-mean ratio: {round(100*(rmse/mean(test)), 1)}%
    ------------------------------------"))
  } 
  
  #create title reporting custom model specification
  str_index <- function (forecast, start=NULL, stop=NULL, int=TRUE) {
    #i=14,16,18 are standard seasonal parameters for forecast$method with ARIMA
    if (int) {
      as.integer(substr(forecast$method, 
                        start=start, 
                        stop=stop))
    } else {
      substr(forecast$method, 
             start=start, 
             stop=stop)
    }
    
  }
  
  if (str_index(forecast, start=1, stop=5, int=F) == "ARIMA") {
    first_s_value <- str_index(forecast, start=14,stop=14, int=F)
    if (first_s_value != "" || !is.na(first_s_value)) {
      #if expanding this for exogenous covariate series, condition "X" as well
      cu.subtitle <- paste("S", sep = "", forecast$method)
    }
  } else {
    cu.subtitle <- forecast$method
  }

  #create and assign table
  eval_tbl <- tibble(
    statistic=c("residuals_mean",
                "residuals_sd",
                "error_mean",
                "error_sd",
                "mae",
                "mape.perc",
                "mse",
                "rmse",
                "test_mean",
                "rmse_test_mean_ratio.perc"),
    values=c(round(residuals_mean),
             round(residuals_sd),
             round(error_mean),
             round(error_sd),
             round(mae),
             round(mape*100),
             round(mse),
             round(rmse),
             round(mean(test)),
             round(100*(rmse/mean(test)), 1))
  )
  colnames(eval_tbl) <- c("Statistic", glue("{cu.subtitle}"))
  
  if (assign.eval_tbl) {
    assign(eval_tbl.name, eval_tbl, envir = .GlobalEnv)
  } #declare user-defined tibble name
  
  #conditioning output for a returned tibble for inline use
  if (return.eval_tbl) {
    return(eval_tbl)
  }
  
  #conditioning output for table to plot area
  require(ggpmisc)
  if (print.eval_tbl) {
    ggplot() + geom_table_npc(data=eval_tbl, label=list(eval_tbl), 
                                       npcx=0.5, npcy=0.5, size=10) + 
      theme(plot.title = element_text(hjust=0.5, vjust=2, size=25))
  }
  
}

# eg)
eval_forecast(ms_ts, forecast, test, train, console=F, return.eval_tbl=F, print.eval_tbl = T)



# Second, write function to fit a model (stick this in the next function or make way to merge with eval_forecast())
fc_fn <- function (ts=ts, train_test_split = TRUE, split_perc=0.85, 
                   fc_len=NULL, assign_fc_obj = c(FALSE, NULL), 
                   eval_fc_output=c("report", "return fc object"),
                   modelvar=c("arima","tbats"),
                   autoarima=FALSE, 
                   autoarima.spec = auto.arima(y = train, max.order = 20, 
                                               max.p = 10, max.d = 3, max.q = 10,
                                               max.P = 10, max.D = 3, max.Q = 10,
                                               stepwise = TRUE, seasonal = TRUE, 
                                               stationary = FALSE,
                                               ic = "aic", trace = TRUE),
                   manual.arima.spec = arima(y = train,
                                             order=c(2,1,0)),
                   manual.tbats.spec = tbats(y = train, trace = TRUE)) { 
  
  # create the training and test sets
  if (train_test_split) {
    train_test_split(ts, split_perc = split_perc)
  } ### DO I NEED THE BOOLEAN HERE? i.e., SHOULDN'T I GIVE OPTION TO INPUT OWN TRAIN/TEST SETS?
  
  # fit the model and create forecast object
  if (modelvar == "arima") {
    if (autoarima == TRUE) {
      model <- autoarima.spec
    } else {
      # for seasonality, include seasonal=list(order=c(w,x,y, period=z) in manual.arima.spec()
      model <- manual.arima.spec
    }
  } else if (modelvar == "tbats") {
    model <- manual.tbats.spec
  }
  #allowing for manual forecast horizon
  if (is.null(fc_len)) {
    forecast <- forecast(model, h = length(test), level = c(80, 95, 99))
  } else {
    forecast <- forecast(model, h = fc_len, level = c(80, 95, 99))
  }
  
  # (optional) assign forecast to GlobalEnv w/ user-defined name
  if (assign_fc_obj[1]) {
    assign(assign_fc_obj[2], forecast, envir = .GlobalEnv)
  }
  
  # (optional) console & plot output, or assign forecast object:
  #options: c(report", "return fc object")
  if (any(eval_fc_output %like% c("report"))) {
    eval_forecast(ts, forecast, test, train, console=T)
    plot_eval_forecast(ts, forecast, test, train, og_df.date_col = ts_df$date)
  } # else if (any(eval_fc_output %like% "return fc object")) {
  #   
  # }
  
}

# eg)
fc_fn(ts, modelvar = "arima", assign_fc_obj = c(TRUE, "fc.1"),
      manual.arima.spec = arima(train,
                                order=c(2,1,6), 
                                seasonal=list(order=c(1,1,1), period=7)),
      eval_fc_output = "report")



#### TESTING THE FUNCTION COMBO in this specific way (i.e., they can both do a lot else):
#first generate 4 forecasts
fc_fn(ts, modelvar = "arima", assign_fc_obj = c(TRUE, "fc.1"),
      manual.arima.spec = arima(train,
                                order=c(2,1,1), 
                                seasonal=list(order=c(1,1,1), period=7)))
fc_fn(ts, modelvar = "arima", autoarima = TRUE, assign_fc_obj = c(TRUE, "fc.2"))
fc_fn(z_ts, modelvar = "arima", autoarima = TRUE, assign_fc_obj = c(TRUE, "fc.3"))
fc_fn(ms_ts, modelvar = "tbats", assign_fc_obj = c(TRUE, "fc.4"))

#second generate 4 tbls (NOTE: fc.1 and tbl.1 need to be run back to back, and so on, to maintain the right train/test sets I think)
eval_forecast(ts, fc.1, test, train,console=F, assign.eval_tbl = T, eval_tbl.name = "tbl.1")
eval_forecast(ts, fc.2, test, train, console=F, assign.eval_tbl = T, eval_tbl.name = "tbl.2")
eval_forecast(z_ts, fc.3, test, train, console=F, assign.eval_tbl = T, eval_tbl.name = "tbl.3")
eval_forecast(ms_ts, fc.4, test, train, console=F, assign.eval_tbl = T, eval_tbl.name = "tbl.4")

#third merge the tables for a ggplot table comparison of model
tbl.final <- tibble(tbl.1, tbl.2[2], tbl.3[2], tbl.4[2])
#tbl.final <- tibble(tbl.1, tbl.2[2], tbl.4[2])
ggplot() + geom_table_npc(data=tbl.final, label=list(tbl.final), 
                          npcx=0.5, npcy=0.5, size=4, 
                          table.theme=ttheme_gtstripes) + theme_minimal() +
  theme(plot.title = element_text(hjust=0.5, vjust=2, size=11))


tbl.1
tbl.2
tbl.3
tbl.4

######################################################################################################
##### TRY THE LSMT MODEL #####
#http://rwanjohi.rbind.io/2018/04/05/time-series-forecasting-using-lstm-in-r/
#http://datasideoflife.com/?p=1171      #this one feels less applicable (Tried it and got strange results but may have a few useful components)

## Preparing the data
# transform data to stationarity
diffed <- diff(ts, differences = 1)

# lagged dataset
lag=365             #ALTERED this from 1 to 365
lag_transform <- function(x, k= lag) {
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
  
}
supervised = lag_transform(diffed, lag)   
head(supervised)

# split into train and test sets
N = nrow(supervised)
n = round(N *0.85, digits = 0)
train = supervised[1:n, ]
test  = supervised[(n+1):N,  ]

## Normalize the data
# scale data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return(list(scaled_train = as.vector(scaled_train), 
              scaled_test = as.vector(scaled_test),
              scaler= c(min =min(x), max = max(x))))
  
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]

# inverse-transform to revert predicted values to og scale
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

## Define the model
# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1)

# specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1                # must be a common factor of both the train and test samples
units = 25                    # can adjust this, in model tuning phase     # have tried 1, 25, & 50

model <- keras_model_sequential() 
model %>%
  layer_lstm(units, 
             batch_input_shape = c(batch_size, 
                                   X_shape2, 
                                   X_shape3), 
             stateful= TRUE) %>%
  layer_dense(units = 1)

# compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)
summary(model)

# fit the model
Epochs = 50                                          # I'm pretty sure the loop is only to "reset_states()" - DROP IT
for(i in 1:Epochs ){
  model %>% fit(x_train, 
                y_train, 
                epochs=1, 
                batch_size=batch_size, 
                verbose=1, 
                shuffle=FALSE)
  #model %>% reset_states()
}

# Make our predictions
L = length(x_test)
scaler = Scaled$scaler
lstm_forecast = numeric(L)

for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  yhat = model %>% 
    predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + ts[(n+i)]
  # store
  lstm_forecast[i] <- yhat
}


## Create forecast object to use with our other functions
# prediction on our training set
L = length(x_train)
fitted = numeric(L)

for(i in 1:L){
  X = x_train[i]
  dim(X) = c(1,1,1)
  yhat = model %>% 
    predict(X, batch_size=batch_size) 
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))     
  # invert differencing
  yhat  = yhat + ts[(i+1)]
  # store
  fitted[i] <- yhat
}

# deal with the prediction length/lag offset at the beginning of series 
#lstm_forecast <- c(rep(NA, n+1), lstm_forecast)
#fitted <- c(rep(NA, 1), fitted) 

# change predicted values to a time series object
lstm_forecast <- ts(lstm_forecast, start=as.Date("2015-05-30"), frequency =1)         
# input series to a ts object
input_ts <- ts(ts_df$sales, start=as.Date("2013-01-01"), frequency = 1)  

# define forecast object
forecast_list <- list(
  model = NULL,
  method = glue("LSTM (memory: {lag})"),
  mean = lstm_forecast,
  x = input_ts,
  fitted = fitted,
  residuals = as.numeric(input_ts) - as.numeric(c(fitted, rep(NA, length(x_test))))
)
class(forecast_list) <- "forecast"


# Plot it
fc_df <- tibble(date = ts_df$date,
                x = forecast_list$x,
                fitted = c(forecast_list$fitted, rep(NA, length(x_test))),
                forecasted = forecast_list$mean)
# y = sales
ggplot(fc_df, aes(x=date)) +
  geom_line(aes(y=x, color="original series"), size=1) +
  geom_line(aes(y=fitted, color="fitted"), size=1) +
  geom_line(aes(y=forecasted, color="forecast"), size=1) +
  scale_color_manual(name="Model",
                     values=c("original series"="black",
                              "fitted"="turquoise4",
                              "forecast"="red"))
# just the predictions + test
ggplot(data=fc_df[length(x_train+1):N,], aes(x=date)) +
  geom_line(aes(y=x, color="test"), size=1) +
  geom_line(aes(y=forecasted, color="prediction"), size=1) +
  scale_color_manual(name="Model",
                     values = c("test" = "black",
                                "prediction" = "red"))


# Try everything out
train_test_split(ts_df$sales, split_perc = 0.85) #have to run again to get the initial sets back 
#had to temporarily change the length statement in the eval_df assignment - FIX LATER
#i.e.,I added +1 to length(test) to compensate for n-1 fitted values. Could possibly correct with an
#NA for the first value if lag=1 then get rid of the +1, or create a parameter to take care of this.
plot_eval_forecast(ts=forecast_list$x, forecast=forecast_list, 
                   test=test, train=train, og_df.date_col = ts_df$date) 
eval_forecast(forecast_list$x, forecast_list, train=train, test=test)

## NOTE: Will need to deep dive a bit further (e.g., youtube, breaking down the python code from initial tutorial).
#
#        This all still needs a bit of work - I don't entirely understand what I'm doing,
#        and whether I'm doing something wrong with the train/test split. I.e., if I am supposed to split
#        the data, and THEN start the code with the training set, leading to a second split into train/test.
#        It looks like I'm using the existing "test" set to make predictions, when I should only be 
#        using it for comparisons to the predicted values. Furthermore, I need to make sure my output columns
#        (e.g., fitted, etc.) have been corrected in their lag. It looks like in some cases values are at t-1.
#
#       
#        
#        Looking at the 4-panel plot from the initial tutorial, it also looks like I'm doing something wrong
#        with my model fitted values, becase I don't think I'm supposed to have any for the first "lagged" 
#        days of the model.
#       
#        All this said, if I am doing things correctly, I've been able to get visually and/or accuracy appealing
#        results by playing with the train/test split, the lag, and the "units".




##################################################################
##THEN OUTLIER REMOVAL/DETRENDING THING for ARIMA

#NB: likely not prudent to do much with the outliers as at least a few of them seem to well represent
#    seasonality (e.g., the anuual spikes around xmas). I.e., they do not appear to be errors (e.g., data 
#    entry), but instead are extreme events or shocks that we'd hope our model could still predict.

## After obtaining "ts_clean" table (vs. ts_df) from premodel program...
# Obtain training & test sets; convert to ts for later use
ms_ts <- msts(ts_clean$sales, seasonal.periods = c(7,365.25))  
z_ts <- zoo(ts_clean$sales)  
ts <- ts(ts_clean$sales, start=as.Date("2013-01-01"), frequency = 7)
train_test_split(ts, split_perc = 0.85, full_df = T, ts_col = ts_clean$sales)

plot_acf_pacf(ts_clean, ci=0.95, diff=T)
# forecast 1: manual arima
fx.1 <- fc_fn(ts, modelvar = "arima", eval_fc_output = "return fc object", 
              manual.arima.spec = arima(train, 
                                        order=c(2,0,1), 
                                        seasonal=list(order=c(0,1,0), 
                                                      period=7)))
plot_eval_forecast(ts, fx.1, train=train, test=test, og_df.date_col = ts_clean$date)
eval_forecast(ts, fx.1, train = train, test = test) #mape: 20%
# forecast 2: autoarima
fx.2 <- fc_fn(ts, modelvar = "arima", eval_fc_output = "return fc object",
              autoarima = TRUE)
plot_eval_forecast(ts, fx.2, train=train, test=test, og_df.date_col = ts_clean$date)
eval_forecast(ts, fx.2, train = train, test = test) #mape: 17%
# forecast 3: tbats only forcing weekly seasonality
fx.3 <- fc_fn(ts, modelvar = "tbats", eval_fc_output = "return fc object")
plot_eval_forecast(ts, fx.3, train=train, test=test, og_df.date_col = ts_clean$date)
eval_forecast(ts, fx.3, train = train, test = test) #mape: 18%
# forecast 4: tbats w/ msts object
train_test_split(ms_ts, split_perc = 0.85, full_df = T, ts_col = ts_clean$sales)
fx.4 <- fc_fn(ms_ts, modelvar = "tbats", eval_fc_output = "return fc object")
plot_eval_forecast(ms_ts, fx.4, train=train, test=test, og_df.date_col = ts_clean$date)
eval_forecast(ms_ts, fx.4, train = train, test = test) #mape: 32%
# forecast 5: arima w/ zoo object (no forced seasonality)
train_test_split(z_ts, split_perc = 0.85, full_df = T, ts_col = ts_clean$sales)
fx.5 <- fc_fn(z_ts, modelvar = "arima", eval_fc_output = "return fc object",
              autoarima = TRUE)
plot_eval_forecast(z_ts, fx.5, train=train, test=test, og_df.date_col = ts_clean$date)
eval_forecast(z_ts, fx.5, train = train, test = test) #mape: 16% (wow... that's depressing)


##################################################################
## THEN YOU"RE DONE WITH THE OVERARCHING CONTENT :)




