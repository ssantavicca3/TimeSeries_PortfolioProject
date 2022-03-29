##### Load R packages

# For Shiny app
library(shiny)
library(shinythemes)
library(shinyWidgets)
library(shinybusy)

# For outlier detection
library(e1071)
library(kernlab)

# For stationarity test and ts decomposition
library(tseries)       #adf test & ts()
library(urca)          #adf test for cv table

# For sample subsetting & forecasting
library(forecast)     #also used for ggAcf and ggPacf plots
library(keras)        #LSTM
library(tensorflow)   #LSTM

# For visualizations
library(ggplot2)
library(ggthemes)
library(highcharter)   #htmlwidgets
library(dygraphs)      #htmlwidgets
library(patchwork)     #multi-panel ggplots
library(RColorBrewer)
library(ggpmisc)       #ggplot tibbles

# For gen. purposes
library(dplyr)         #manipulation
library(magrittr)      #pipe & subsetting secondary series w/in ggplot
library(zoo)           #rolling stats and interpolation; time-series object w/ 'daily' datetime index 
library(glue)          #string magic
library(wrapr)         #pipe ggplot layers
library(stringr)       #string magic
library(data.table)    #%like%


##### Write or amend certain operators

# Create "%notin%" operator by negating "%in%"
`%notin%` <- Negate(`%in%`)

# Rewrite (wrapr) dot-arrow-pipe S3 dispatch rules to pipe ggplot:
apply_left.gg <- function(pipe_left_arg,
                          pipe_right_arg,
                          pipe_environment,
                          left_arg_name,
                          pipe_string,
                          right_arg_name) {
  pipe_right_arg <- eval(pipe_right_arg,
                         envir = pipe_environment,
                         enclos = pipe_environment)
  pipe_left_arg + pipe_right_arg 
}
apply_right.gg <- function(pipe_left_arg,
                           pipe_right_arg,
                           pipe_environment,
                           left_arg_name,
                           pipe_string,
                           right_arg_name) {
  pipe_left_arg + pipe_right_arg 
}
apply_right.labels <- function(pipe_left_arg,
                               pipe_right_arg,
                               pipe_environment,
                               left_arg_name,
                               pipe_string,
                               right_arg_name) {
  if(!("gg" %in% class(pipe_left_arg))) {
    stop("apply_right.labels expected left argument to be class-gg")
  }
  pipe_left_arg + pipe_right_arg 
}


##### Assign common ggplot elements to be subbed later on
theme_standard <- theme_minimal() +
  theme(plot.title = element_text(hjust=0.5, size=20, face="bold"))
theme_4panel <- theme_minimal() +
  theme(axis.text.x = element_blank(), axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
        plot.margin = unit(c(0,0,0,0), "cm"),
        panel.border = element_rect(color="black", fill=NA, size=.5))
labels_standard <- labs(x = "", y = "Sales ($)", title = "Total Daily Sales (2013-15)")
vline_outliers <- geom_vline(xintercept = 10000, color="red", linetype="dashed")
date_breaks_4panel <- scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                                      "2014-01-01", "2014-05-01", "2014-09-01",
                                                      "2015-01-01", "2015-05-01", "2015-09-01")))

# theme for forecast data objects
theme.fxdat <- theme_gdocs() +
  theme(plot.title = element_text(size = 15),
        plot.subtitle = element_text(size = 11),
        plot.caption = element_text(size = 9, hjust = 0, vjust = 0, colour = "grey50"),
        axis.title.y = element_text(face = "bold", color = "gray30"),
        axis.title.x = element_text(face = "bold", color = "gray30", vjust = -1),
        panel.background = element_rect(fill = "grey95", colour = "grey75"),
        panel.border = element_rect(colour = "grey75"),
        panel.grid.major.y = element_line(colour = "white"),
        panel.grid.minor.y = element_line(colour = "white", linetype = "dotted"),
        panel.grid.major.x = element_line(colour = "white"),
        panel.grid.minor.x = element_line(colour = "white", linetype = "dotted"),
        strip.background = element_rect(size = 1, fill = "white", colour = "grey75"),
        strip.text.y = element_text(face = "bold"),
        axis.line = element_line(colour = "grey75"))


##### Import Data

df <- read.csv("Data/sales_train.csv")

## Create "sales" series with a daily frequency datetime index
# Format datetime column
df$date <- as.Date(df$date, "%d.%m.%Y")

# Create time series
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


##### User-Defined Functions

## Function to create a trends plot with rolling statistics (mean, CI)
trendy_plot <- function (ts_df, plot_ma=TRUE, plot_intervals=TRUE, window=5) {
  
  ts <- ts_df
  
  rolling_avg <- zoo::rollmean(ts$sales, k=window, align="right")
  rolling_std <- zoo::rollapply(ts$sales, width=window, sd, align="right")
  
  lower_bound <- rolling_avg - (1.96*rolling_std)
  upper_bound <- rolling_avg + (1.96*rolling_std)
  
  #subset original series to match window length
  ts_fun <- ts[window:length(ts$sales),]
  
  p <- ts_fun %>%
    data.frame(rolling_avg, lower_bound, upper_bound) %>% 
    ggplot(aes(x = date)) +
    geom_line(aes(y=sales), size=.75) +
    theme_standard +
    labels_standard # %.>%
  # scale_color_manual(name = "Sales", 
  #                    values = "turquoise4",
  #                    labels = "Actual Values")
  
  if (plot_ma) {
    p <- p + geom_line(aes(y = rolling_avg, color = 'red'), size=.75) +
      scale_color_manual(name = "", 
                         values = c("red", "turquoise4"),
                         labels = c("Rolling Avg.", "Actual Values"))
  }
  
  if (plot_intervals) {
    p <- p + geom_ribbon(aes(x=date, ymax=upper_bound, ymin=lower_bound),
                         fill="grey70", alpha=.4) 
  }
  
  print(p)
  
}


## Function to detect outliers using SVM
detect_outliers <- function (ts_df, perc=0.01, gamma=0.01, return_df=TRUE, plot_ts=TRUE) {
  
  ts <- ts_df
  
  # train a one-class SVM model
  model <- ksvm(ts$sales, nu=perc, type='one-svc', kernel='rbfdot', 
                kpar=list((sigma=gamma))) #using radial bias f'n kernel
  
  # time series output
  ts_outliers <- data.frame(ts)
  ts_outliers$index <- 0 #initializing
  
  for (i in 0:length(as.vector(ts$sales))) {
    ts_outliers$index[i] = i
  }
  
  ts_outliers$outlier <- predict(model)
  
  for (i in 1:length(as.vector(ts$sales))) {
    if (ts_outliers$outlier[i] == TRUE) {
      ts_outliers$outlier[i] = 0
    }
    else {
      ts_outliers$outlier[i] = 1
    }
  }
  
  # plot ts w/ exposed outliers
  n_outliers <- ts_outliers %>% count(outlier)
  
  p <- ggplot(ts_outliers) +
    geom_line(aes(x = index, y = sales, 
                  color = "turquoise4"), size=1) +
    geom_point(aes(x = index, y = sales), 
               data = . %>% filter(outlier %in% 1), color = 'red', size=3, alpha=0.5) +
    labs(x="", y="", title=glue("Outliers Detected: Found {n_outliers[2,2]}")) +
    scale_color_manual(name="",
                       values = c("turquoise4", "red"),
                       labels = c("Actual Values", "Outliers")) +
    theme(legend.justification = c(1,0), legend.position=c(.95,.9))
  
  # conditional function output
  if (plot_ts) {
    print(p)
  }
  
  if (return_df) {
    return(ts_outliers)
  }
  
}


## Function to interpolate outliers for removal
remove_outliers <- function (ts_df, outliers_idx, ts_outliers, return_df = TRUE, plot_ts = TRUE) {
  
  ts <- ts_df
  
  ts_clean <- ts
  ts_clean$sales[outliers_idx] <- NA
  ts_clean$sales <- na.approx(ts_clean$sales, method='linear') #interpolation
  
  # plot the two series
  ts_clean$og_sales <- ts$sales
  n_outliers <- ts_outliers %>% count(outlier)
  
  p <- ggplot(ts_clean) +
    geom_line(aes(x=date, y=og_sales, color="red"), size=1) +
    geom_line(aes(x=date, y=sales, color="turquoise4"), size=1) +
    labs(x="Date", y="Sales", 
         title=glue("Outliers Removed: Found {n_outliers[2,2]}")) +
    scale_color_manual(name = "Sales", 
                       values = c("red", "turquoise4"),
                       labels = c("Original", "Interpolated")) 
  
  # conditional function output
  if (plot_ts) {
    print(p)
  }
  
  if (return_df) {
    return(ts_clean)
  }
  
}


## Function to visualize stats and partial/autocorrelation and run ADF test
plot_stationarity_test <- function (ts_df, sample=0.20, maxlag=30) {
  
  ts <- ts_df
  
  # test stationarity (Augmented Dickey-Fuller test)
  test <- adf.test(ts$sales, k=maxlag)
  adf <- test$statistic
  pval <- test$p.value
  lag_order <- test$parameter
  conclusion <- if (pval < 0.05) "Stationary" else "Non-Stationary"
  pval_edt <- if(pval <= 0.01) "<= 0.01" else round(pval, 3)
  
  # obtain critical values table from urca::ur.df adf test (same specs)
  ts$sales %.>% 
    urca::ur.df(., lags = maxlag, type = "trend") %.>%                   
    { cv_table <- summary(.) } %.>%
    { cv_95 <- .@cval['tau3', '5pct'] }
  cv_99 <- cv_table@cval['tau3', '1pct']
  
  # plot ts with mean/std of a sample from the first x% and report on ADF
  dtf_ts <- ts
  sample_size <- as.integer(length(ts$sales)*sample)
  dtf_ts$mean <- mean(head(dtf_ts$sales, sample_size))
  dtf_ts$upper <- mean(head(dtf_ts$sales, sample_size)) +
    sd(head(dtf_ts$sales, sample_size))
  dtf_ts$lower <- mean(head(dtf_ts$sales, sample_size)) -
    sd(head(dtf_ts$sales, sample_size))
  
  p <- ggplot(dtf_ts, aes(x=date)) +
    geom_ribbon(aes(ymax=upper, ymin=lower),
                fill="grey70", alpha=.4) +
    geom_line(aes(y=sales, color = "turquoise4")) + 
    geom_line(aes(y=mean, color = "red")) +
    labs(x="Date", y="", title=glue("Augmented Dickey-Fuller Test: {conclusion} (p-value: {pval_edt})")) +
    scale_color_manual(values = c("red", "turquoise4")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold")) +
    theme(legend.position = "none") +
    scale_x_date(breaks = as.Date(c("2013-01-01", "2013-07-01", "2014-01-01", "2014-07-01", "2015-01-01","2015-07-01")),
                 labels=c("Jan. 2013", "Jul", "Jan. 2014", "Jul", "Jan. 2015", "Jul")) +
    annotate("text", x=as.Date("2015-07-01"), y=10500, 
             label=glue("ADF Stat: {round(adf,2)}"))
  
  # pacf (for AR) and acf (for MA)
  lower_ylim <- min(min(acf(ts$sales, plot=F)$acf),
                    min(pacf(ts$sales, plot=F)$acf))   #automating graph alignment
  acf_plot <- ggAcf(ts$sales, lag.max=maxlag, ci=0.95) + 
    ylim(c(lower_ylim, 1)) +
    geom_point(aes(x=lag, y=Freq), color="blue", shape=21, size=1.5, stroke=1.5, fill="white") +
    labs(y="", title="Autocorrelation (for MA component)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold")) 
  pacf_plot <- ggPacf(ts$sales, lag.max=maxlag, ci=0.95) + 
    ylim(c(lower_ylim, 1)) +
    geom_point(aes(x=lag, y=Freq), color="blue", shape=21, size=1.5, stroke=1.5, fill="white") +
    labs(y="", title="Partial Autocorrelation (for AR component)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold"))
  
  #arrange plots with patchwork::
  p / (acf_plot + pacf_plot) 
  
}


## Function to split ts and output train and test sets (can be used in-line)
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


## Function comparing the rmspe to the test mean
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
    theme(plot.subtitle = element_text(lineheight = 0.55))   #closing gap b/w subtitles +
  
}


## Function to render 4-panel plot for forecast evaluation
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
  # deeper near-burgundy red: #CC3333
  # brighter off-red: #FF6666
  
}


## Function to evaluate algorithm performance
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


# Function to fit a model and produce forecast-object based output with other user-defined functions
fc_fn <- function (ts=ts, train_test_split = TRUE, split_perc=0.85, 
                   fc_len=NULL, assign_fc_obj = c(FALSE, NULL), 
                   eval_fc_output=c("report", "return eval object"),
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
    train <- train_test_split(ts, split_perc = split_perc, out.train = T)
    test <- train_test_split(ts, split_perc = split_perc, out.test = T)
  } ### DO I NEED CONDITON HERE? i.e., SHOULDN'T I GIVE OPTION TO INPUT OWN TRAIN/TEST SETS?
  
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
  } else if (any(eval_fc_output %like% "return eval object")) {
    eval_tbl <- eval_forecast(ts, forecast, test, train, console=F, return.eval_tbl = T)
    return(eval_tbl)
  }
  
}

####################################################################################


# source : https://shiny.rstudio.com/gallery/ncaa-swim-team-finder.html


##### Define UI
ui <- fluidPage(
  theme = shinytheme("superhero"),
  navbarPage(
    "Time Series ML Forecasting",
    navbarMenu("Intro", icon = icon("info-circle"),
               tabPanel("About", fluid = TRUE,
                        fluidRow(
                          h3(p("About the Project"))
                        ),
                        fluidRow(
                          column(6,
                                 tags$b(h4("BACKGOUND")),
                                 p(style="text-align: justify;",
                                  "I have been utilizing R sporadically since 2016 for various work and academic projects, but I incurred 
                                   several months without doing so, and several more learning Python. Additionally, I had only scraped the 
                                   surface of forecasting methods during my secondary and post-seconday education in economics, 
                                   yet the tool set was one that greatly interested me."),
                                 tags$b(h4("INSPIRATION")),
                                 p(style="text-align: justify;",
                                   "I stumbled across a great read on Towards Data Science by Mauro Di Pietro titled 'Time Series Analysis for 
                                   Machine Learning.' In this article, Mauro follows a typical time-series analysis of a dataset from the Kaggle
                                   competition 'Predict Future Sales' and uses Python as his language of choice.")
                          ),
                          column(6,
                                 tags$b(h4("MY GOALS")),
                                 tags$ul(
                                   tags$li("Brush up on and further develop my skills in R statistical programming while reinforcing new skills in Python"),
                                   tags$li("Properly introduce myself to time series analysis and ML forecasting."),
                                   tags$li("Experiment with different plot methods and themes."),
                                   tags$li("Automate certain forecasting analysis processes by manipulating R forecasting objects."),
                                   tags$li("Create interactive web application to allow users to access the full process, from 
                                           time series analysis to model design and testing for forecasting.")
                                 )
                          )
                        ),
                        hr(),
                        panel(
                          tags$b(h4("SOURCES")),
                          tags$ul(
                            tags$li(tags$a(href="https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205",
                                           tags$i("Time Series Analysis for Machine Learning"))),
                            tags$li(tags$a(href="https://www.kaggle.com/c/competitive-data-science-predict-future-sales",
                                           "Predict Future Sales, Kaggle Competition")),
                            tags$li(tags$a(href="https://rh8liuqy.github.io/ACF_PACF_by_ggplot2.html",
                                           "ACF and PACF Correlogram, rh8liugy, github")),
                            tags$li(tags$a(href="http://applied-r.com/plotting-forecast-data-objects-ggplot/",
                                           "Custom ggplot theme for forecasting data objects, Brad Horn, applied-R"))
                          )
                        )
                        
               ), # tabpanel
              tabPanel("Overview", fluid = TRUE,
                       fluidRow(
                         h3(p("Project Overview")),
                       ),
                       fluidRow(
                         plotOutput("plt_ts")
                       ),
                       fluidRow(
                         h4("INSTRUCTIONS"),
                         p(style="text-align: justify;",
                           "This web application allows you to experiment and interact with each step of the forecasting process."),
                         p(style="text-align: justify;",
                           "First you will explore the different stages of a standard time series analysis. This includes trend analysis, 
                           outlier dection (and treatment), stationarity testing, and time series decomposition."),
                         p(style="text-align: justify;",
                           "Next, you'll build models and assess their prediction accuracy. You can manually fit (S)ARIMA and (T)BATS models, 
                           but first, you must select your train-test split percentage. This is the percentage of the initial time series you 
                           will use to train the model, whereas the remainder will be used to test your models' performance. While under the 
                           'Time Series Analysis' dropdown menu, you can always return to the 'Train-Test Split & Dry Forecast' tab to
                           adjust your selection for this threshold. In this tab, try auto-fitting the SARIMA model (using stepwise selection)
                           with your chosen train-test percentage. When assessing prediction accuracy, the ratio of root mean squared error 
                           to mean (rmse-mean ratio) is a good metric to try to minimize, albiet this alone does not reveal the full picture."),
                         p(style="text-align: justify;",
                           "Finally, you will visualize your model's performance in the last stage of this application. The 'Model Performance tab' 
                           stands on its own; meaning, you can independently declare a train-test split percentage in this section when building 
                           your models. After doing so, and specifying values parameter values for your model, hit 'Run' to render an array of 
                           accuracy charts to visualize your model's performance. Full screen is recommended for best layout for this page."),
                         hr(),
                         panel(
                           h4("OUTLINE"),
                           column(4,
                                  h4("Step 1: Time Series Analysis"),
                                  tags$ul(
                                    tags$li(p("Trend Analysis")),
                                    tags$li(p("Outlier Dection")),
                                    tags$li(p("Stationarity Test")),
                                    tags$li(p("Time Series Decomposition"))
                                  )
                           ),
                           column(4,
                                  h4("Step 2: Model Design and Testing"),
                                  tags$ul(
                                    tags$li(p("Train-Test Split and Dry Forecast")),
                                    tags$li(p("Manually Fit (S)ARIMA Model")),
                                    tags$li(p("Manually Fit (T)BATS Model"))
                                  )
                           ),
                           column(4,
                                  h4("Step 3: Model Performance")
                           )
                         )
                         
                       )
              ) # tabPanel
    ), # navbarMenu
    
    navbarMenu("Time Series Analysis", icon = icon("balance-scale-right"),
               tabPanel("Trends", fluid = TRUE,
                        titlePanel("Trend Analysis"),
                        sidebarPanel(
                          sliderInput(inputId = "slider_trendanalysis", 
                                      label = h3(HTML(paste0("Window length ", "<i>", "k", "</i>", " (days)"))), 
                                      min = 1, max = 365, 
                                      value = 30),
                          radioButtons(inputId = "radio_trendanalysis", 
                                       label = h3("Rolling Statistics"),
                                       choices = c("Rolling Average",
                                                   "Bollinger Bands",
                                                   "Both"))
                        ),
                        mainPanel(
                          plotOutput("trends_plot")
                        )
                          
               ), # tabpanel
               
               tabPanel("Outliers", fluid = TRUE,
                        titlePanel("Outlier Detection"),
                        sidebarLayout(
                          sidebarPanel(
                            radioButtons(inputId = "radio_outliers1", 
                                         label = h3("Distribution Plots"),
                                         choices = c("Histogram",
                                                     "Density Function",
                                                     "Box & Whiskers")),
                            hr(),
                            titlePanel("Support Vector Machine"),
                            knobInput(inputId = "slider_outliers",
                                      label = h3("Threshold (%)"),
                                      min = 1, max = 100,
                                      value = 5,
                                      displayPrevious = TRUE,
                                      lineCap = "round",
                                      fgColor = "#428BCA",
                                      inputColor = "#428BCA"),
                            # sliderInput(inputId = "slider_outliers",
                            #             label = h3("Threshold (%)"),
                            #             min = 0.0, max = 1.0,
                            #             value = 0.05),
                            radioButtons(inputId = "radio_outliers2",
                                         label = h3("Treatment of Outliers"),
                                         choices = c("Detect", 
                                                     "Remove")),
                            helpText(style="text-align: justify;",
                                     "Note: Algorithm can detect outliers at any threshold,
                                     however, removing them using a threshold > 15% will
                                     break it.")
                          ),
                          mainPanel(
                            plotOutput("outlier_dist_plot"),
                            plotOutput("outlier_svm_plot")
                          )
                        )
                        
               ), # tabPanel
               
               tabPanel("Stationarity", fluid = TRUE,
                        titlePanel("Augmented Dickey-Fuller Test"),
                        sidebarLayout(
                          sidebarPanel(
                            sliderInput(inputId = "slider_station1",
                                        label = h3("Sample (%)"),
                                        min = 0.0, max = 1.0,
                                        value = 0.2),
                            sliderInput(inputId = "slider_station2",
                                        label = h3("Max lag (days)"),
                                        min = 0, max = 365,
                                        value = 30),
                            hr(),
                            titlePanel("Stabilize the Mean"),
                            helpText(HTML(paste0("Differencing the mean with ", "<i>", "lag", "</i>", " = 1 day."))),
                            materialSwitch(inputId = "switch_station",
                                           status = "info",
                                           label = "Apply lag")
                          ),
                          mainPanel(
                            plotOutput("station_test_plot"),
                          )
                        )
                        
               ), # tabpanel
               tabPanel("Seasonality", fluid = TRUE,
                        titlePanel("Time Series Decomposition"),
                        sidebarPanel(
                          checkboxGroupInput(inputId = "checkbox_season",
                                             label = h3("Series Components"),
                                             choices = c("Original",
                                                         "Trend",
                                                         "Seasonal",
                                                         "Residual"),
                                             selected = "Residual"),
                          # helpText(h2("NOTE TO SELF: the checkboxgroups would probably look better sitting above the plot vs. to the left."))
                        ),
                        mainPanel(
                          plotOutput("season_plot")
                        )
                        
               ) # tabPanel
               
    ), # Time Series Analysis, navbarMenu
    
    navbarMenu("Model Design & Testing", icon = icon('chart-bar'),
               tabPanel("Train-Test Split & Dry Forecast", fluid = TRUE,
                        titlePanel("Train-Test Split"),
                        fluidRow(
                          sidebarPanel(
                            h3("Instructions:"),
                            helpText(style="text-align: justify;",
                                     p("Begin by selecting the percentage of the time series that 
                                       you would like to devote to training your model."),
                                     p("Then flick the 'Forecast' switch to run a dry forecast of daily sales
                                       by fitting an ARIMA model using stepwise selection (autoarima() in R).")),
                            hr(),
                            h5("Notes:"),
                            helpText(style="text-align: justify;",
                                     p("- This percentage will be saved and appliedto the models 
                                       you will build in the next two tabs of the 'Model Design & 
                                       Testing' drop-down menu."),
                                     p("- You can return to this tab at any point to adjust this 
                                       percentage.")),
                            hr(),
                            sliderInput(inputId = "slider_traintest",
                                        label = h3("Training set (%)"),
                                        min = 1, max = 100,
                                        value = 85),
                            hr(),
                            materialSwitch(inputId = "switch_traintest",
                                           status = "info",
                                           label = h3("Forecast")),
                            helpText("Note: The stepwise algorithm for autoarima() will take up to 30 seconds to complete iterations for most users."),
                            width = 2
                          ),
                          mainPanel(
                            fluidRow(
                              panel(
                                plotOutput("traintest_plot1"),
                                plotOutput("traintest_plot2")
                              )
                            )
                          )
                        )
               ), # tabPanel
               
               tabPanel("Manually Fit (S)ARIMA MODEL", fluid=T,
                        titlePanel("(S)ARIMA Model Specification"),
                        sidebarLayout(
                          sidebarPanel(
                            helpText(style="text-align: justify;",
                                     "If you have not yet selected a % of the initial data series to use for 
                                     a training set, please return to the tab titled 'Train-Test Split & Dry Forecast' 
                                     under the 'Model Design & Testing' dropdown menu to choose your split %."),
                            hr(),
                            h4("Non-Seasonal Components"),
                            numericInput(inputId = "ar_nonseason",
                                         label="Autoregression order (p):",
                                         value=0),
                            numericInput(inputId = "diff_nonseason",
                                         label="Degree of differencing (d):",
                                         value=0),
                            numericInput(inputId = "ma_nonseason",
                                         label="Moving average order (q):",
                                         value=0),
                            h4("Seasonal Components"),
                            numericInput(inputId = "ar_season",
                                         label="Autoregression order (P):",
                                         value=0),
                            numericInput(inputId = "diff_season",
                                         label="Degree of differencing (D):",
                                         value=0),
                            numericInput(inputId = "ma_season",
                                         label="Moving average order (Q):",
                                         value=0),
                            numericInput(inputId = "period_season",
                                         label="Seasonal period (m):",
                                         value=0),
                            #hr()
                            # helpText("Something informative here that can also fill out the space... OR
                            #          Paste the image of the SARIMA breakdown I saved to the folder using one
                            #          of these techniques: https://shiny.rstudio.com/articles/images.html")
                          ),
                          mainPanel(
                            #Put the "accuracy of algorithm" & "test vs. prediction" plots here
                            verbatimTextOutput("eval_alg_arima"),
                            hr(),
                            plotOutput("arima_plot1")
                          )
                        ), # sidebarLayout
               ), # tabPanel
               
               tabPanel("Manually Fit (T)BATS MODEL", fluid=T,
                        titlePanel("(T)BATS Model Specification"),
                        sidebarLayout(
                          sidebarPanel(
                            helpText(style="text-align: justify;",
                                     "If you have not yet selected a % of the initial data series to use for 
                                     a training set, please return to the tab titled 'Train-Test Split & Dry Forecast' 
                                     under the 'Model Design & Testing' dropdown menu to choose your split %."),
                            hr(),
                            selectInput(inputId = "boxcox_tbat",
                                        label="Box-Cox transformation:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE"),
                            selectInput(inputId = "trend_tbats",
                                        label="Trend:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE"),
                            selectInput(inputId = "trendDP_tbats",
                                        label="Trend dampening parameter:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE"),
                            selectInput(inputId = "armaErrors_tbats",
                                        label="ARMA errors:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE"),
                            hr(),
                            materialSwitch(inputId = "switch_tbats",
                                           status = "info",
                                           label = "Auto-fit model*"),
                            helpText(style="text-align: justify;",
                                     "*If 'Auto-fit model' is selected then the algorithm will fit
                                     the model with and without the parameters in question
                                     and the 'best fit' is chosen by AIC.")
                          ),
                          mainPanel(
                            #Put the "accuracy of algorithm" & "test vs. prediction" plots here
                            verbatimTextOutput("eval_alg_tbats"),
                            hr(),
                            plotOutput("tbats_plot1")
                          )
                        ), # sidebarLayout
               ), # tabPanel
               
    ), # Model Desing & Testing, navbarMenu
    

    tabPanel("Model Performance", fluid = TRUE, icon = icon('chart-line'),
             titlePanel("Specify Your Model"),
             fluidRow(
               sidebarPanel(
                 h3("Instructions:"),
                 helpText(style="text-align: justify;",
                          p("First, select the percentage of the time series that you would
                             like to devote to training your model."),
                          p("Next, enter values for the forecasting
                             model parameters (i.e., specify your model.)"),
                          p("Finally, hit the 'Run' switch to 
                             visualize your model's performance.")),
                 hr(),
                 sliderInput(inputId = "slider_traintest2",
                             label = h4("Training set (%)"),
                             min = 1, max = 100,
                             value = 85),
                 width=2
               ),
               mainPanel(
                 fluidRow(
                   panel(
                     column(2,
                       h3("(S)ARIMA", style="text-align: center;"),
                       helpText("Please specify your model", style="text-align: center;")
                     ),
                     column(2,
                            numericInput(inputId = "ar_nonseason2",
                                         label="AR (p):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "diff_nonseason2",
                                         label="Diff (d):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "ma_nonseason2",
                                         label="MA (q):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "ar_season2",
                                         label="AR (P):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "diff_season2",
                                         label="Diff (D):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "ma_season2",
                                         label="MA (Q):",
                                         value=0)),
                     column(2,
                            numericInput(inputId = "period_season2",
                                         label="Period (m):",
                                         value=0))
                   ),
                 ),
                 hr(),
                 fluidRow(
                   panel(
                     column(2,
                       h3("(T)BATS", style="text-align: center;"),
                       helpText("Please specify your model", style="text-align: center;")
                     ),
                     column(2,
                            selectInput(inputId = "boxcox_tbat2",
                                        label="Box-Cox transformation:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE")),
                     column(2,
                            selectInput(inputId = "trend_tbats2",
                                        label="Trend:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE")),
                     column(2,
                            selectInput(inputId = "trendDP_tbats2",
                                        label="Trend dampening parameter:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE")),
                     column(2,
                            selectInput(inputId = "armaErrors_tbats2",
                                        label="ARMA errors:",
                                        choices = list("TRUE",
                                                       "FALSE"),
                                        selected = "FALSE"))
                   )
                 ),
                 width=10
               )
             ),
             titlePanel("Visualize Model Performance"),
             fluidRow(
               splitLayout(cellWidths = c("50%", "50%"),
                           plotOutput("tbats_plot2"), 
                           plotOutput("arima_plot2"))
             )
             
    ) # Model Performance, tabPanel
    
  ) # navbarPage
  
) # fluidPage


####################################################################################
##### Define server function
server <- function(input, output, session) {
  
  ### Intro ()
  
  # basic ts plot
  output$plt_ts <- renderPlot({
    ggplot(ts_df, aes(x=date, y=sales)) +
      geom_line(color="black", size=.75) +
      theme_minimal() +
      labs(x="", y="", title="Total Daily Sales, USD (2013-15)") +
      theme(plot.title = element_text(hjust=0.5, size=20, face="bold")) 
  })
  
  
  ### Navbar 1
  
  output$txtout <- renderText({
    paste(input$txt1, input$txt2, sep = " ")
  })
  
  ### Time Series Analysis
  
  ## Trends
  
  output$trends_plot <- renderPlot({
    trendy_plot(ts_df, 
                window = input$slider_trendanalysis,
                plot_ma = (input$radio_trendanalysis %in% c("Rolling Average", "Both")),
                plot_intervals = (input$radio_trendanalysis %in% c("Bollinger Bands", "Both")))
  })
  
  ## Outliers
  
  # Distribution Plots
  output$outlier_dist_plot <- renderPlot({
    if (input$radio_outliers1 == "Histogram") {
      # Plot a simple histogram & PDF/CDF
      ts_df %>%
        ggplot(aes(sales)) +
          geom_histogram(color="turquoise4", fill="lightgreen", alpha=.5, bins=100) +
          theme_standard +
          vline_outliers +
          labs(x="Sales ($)", y="Frequency", title="Daily Sales Distribution")
    } else if (input$radio_outliers1 == "Density Function") {
      # Plot a simple CDF/PDF
      options(scipen=10000)
      ts_df %>%
        ggplot(., aes(sales)) + 
        geom_density(color="turquoise4", fill="lightgreen", alpha=.4) +  
        theme_standard + 
        vline_outliers + 
        labs(x="Sales ($)", y="Density", title="Daily Sales Density Function")
    } else {
      # Plot a simple boxplot
      ts_df %>% 
        ggplot(., aes(sales)) +  
        geom_boxplot(color="turquoise4", fill="lightgreen",
                     alpha=.4, outlier.shape = 1, outlier.color = "red") + 
        vline_outliers + 
        theme_standard + 
        theme(axis.ticks = element_blank(), axis.text.y = element_blank()) +  
        labs(x="Sales ($)", y="", title="B&W Plot of Daily Sales")
    }
  })
  
  # Detect/Remove Outliers
  slider_outliers_react <- reactive ({
    input$slider_outliers/100
  })
  output$outlier_svm_plot <- renderPlot({
    if (input$radio_outliers2 == "Remove") {
      # detect outliers
      ts_outliers_react <- reactive({
        detect_outliers(ts_df, perc = slider_outliers_react())
      })
      # outliers' index position
      outliers_index_pos_react <- reactive({
        ts_outliers_react()[ts_outliers_react()$outlier == 1, 3] #'3' is the index column we created
      })
      # exclude outliers
      ts_clean <- remove_outliers(ts_df, ts_outliers = ts_outliers_react(), outliers_idx = outliers_index_pos_react())
    } else {
      detect_outliers(ts_df, return_df = FALSE,
                      gamma = 0.01,
                      perc = slider_outliers_react())
    }
  })
  
  ## Stationarity Test
  
  # Augmented Dickey-Fuller Test
  output$station_test_plot <- renderPlot({
    if (!input$switch_station) {
      plot_stationarity_test(ts_df, 
                             sample = input$slider_station1, 
                             maxlag = input$slider_station2)
    } else {
      # Stabilize the mean by differencing the ts
      lag_ts_react <- reactive({
        lag_ts_pre <- ts_df %>% 
          mutate_all(lag,n=1)
        lag_ts_pre$sales <- ts_df$sales - lag_ts_pre$sales
        lag_ts_pre$date <- ts_df$date
        lag_ts <- lag_ts_pre[rowSums(is.na(lag_ts_pre))==0,]
      })
      plot_stationarity_test(lag_ts_react(), 
                             sample = input$slider_station1,
                             maxlag = input$slider_station2)
    }
  })
  
  ## Seasonality/Decomposition
  
  ts_df$sales %.>% 
    { units <- ts(., frequency = 7) } %.>%   #weekly seasonality
    { decomp <- reactive({
        stl(., s.window='periodic')
    }) } 
  
  # Original Plot
  original_plt_react <- reactive({
    ggplot(ts_df, aes(x=date, y=sales)) +
      geom_line(color="#006699") +
      theme_4panel +
      labs(title="Original Series") +
      date_breaks_4panel
  })
  
  # Trend Plot
  trend_plt_react <- reactive({
    decomp() %.>%
      { .$time.series[,2] } %.>%
      { trend <- cbind(data.frame(.), ts_df$date) } %.>%
      { colnames(trend) <- c("trend", "date") } %.>%
        ggplot(trend, aes(x=date, y=trend)) %.>%
          geom_line(color="#006699") %.>%
          theme_4panel %.>%
          labs(title="Trend") %.>%
          date_breaks_4panel
  })
  
  # Seasonal Plot
  seasonal_plt_react <- reactive({
    decomp() %.>%
      { .$time.series[,1] } %.>%
      { seasonal <- cbind(data.frame(.), ts_df$date) } %.>%
      { colnames(seasonal) <- c("seasonal", "date") } %.>%
        ggplot(seasonal, aes(x=date, y=seasonal)) %.>%
          geom_line(color="#006699") %.>%
          theme_4panel %.>%
          labs(title="Seasonal") %.>%
          date_breaks_4panel
  })
  
  # Residual Plot
  residual_plt_react <- reactive({ 
    decomp() %.>%
      { .$time.series[,3] } %.>%
      { residual <- cbind(data.frame(.), ts_df$date) } %.>%
      { colnames(residual) <- c("residual", "date") } %.>%
        ggplot(residual, aes(x=date, y=residual)) %.>%
          geom_line(color="#006699") %.>%
          theme_minimal() %.>%
          labs(title="Residual") %.>%
          theme(axis.title.y = element_blank(),
                plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
                plot.margin = unit(c(0,0,0,0), "cm"),
                panel.border = element_rect(color="black", fill=NA, size=.5)) %.>%
          scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                          "2014-01-01", "2014-05-01", "2014-09-01",
                                          "2015-01-01", "2015-05-01", "2015-09-01")),
                       date_labels = "%Y-%m")   #unfortunately can't get date_breaks = "4 months" to pull correct labels
  })
    
  
  
  # output$season_plot <- renderPlot({
  #   if (input$checkbox_season %in% all(c("Original","Trend","Seasonal","Residual"))) {
  #     # combined figure using patchwork::
  #     original_plt_react() /
  #     trend_plt_react() /
  #     seasonal_plt_react() /
  #     residual_plt_react()
  #   } else if (input$checkbox_season %in% "Trend") {
  #       trend_plt_react()
  #   } else if (input$checkbox_season %in% "Seasonal") {
  #       seasonal_plt_react()
  #   } else if (input$checkbox_season %in% "Residual") {
  #       residual_plt_react()
  #   } else {
  #       original_plt_react()
  #   }
  # })

  output$season_plot <- renderPlot({
    seasonPlot_checkbox_filter <- function(x) {
      op_cnt = 0
      tp_cnt = 0
      sp_cnt = 0
      rp_cnt = 0
      for (i in 1:length(x)) {
        if (x[i] == "Original") {
          op <- original_plt_react()
          op_cnt <- op_cnt + 1
        } else if (x[i] == "Trend") {
          tp <- trend_plt_react()
          tp_cnt <- tp_cnt + 1
        } else if (x[i] == "Seasonal") {
          sp <- seasonal_plt_react()
          sp_cnt <- sp_cnt + 1
        } else if (x[i] == "Residual") {
          rp <- residual_plt_react()
          rp_cnt <- rp_cnt + 1
        }
      }
      # Plot chart combinations
      if(length(x) == 4) original_plt_react()/trend_plt_react()/seasonal_plt_react()/residual_plt_react() else
      if(length(x) == 3 & op_cnt == 0) trend_plt_react()/seasonal_plt_react()/residual_plt_react() else
        if(length(x) == 3 & sp_cnt == 0) original_plt_react()/trend_plt_react()/residual_plt_react() else
          if(length(x) == 3 & tp_cnt == 0) original_plt_react()/seasonal_plt_react()/residual_plt_react() else
            if(length(x) == 3 & rp_cnt == 0) original_plt_react()/seasonal_plt_react()/trend_plt_react() else
      if(length(x) == 2 & op_cnt == 1 & tp_cnt == 1) original_plt_react()/trend_plt_react() else
        if(length(x) == 2 & op_cnt == 1 & sp_cnt == 1) original_plt_react()/seasonal_plt_react() else
          if(length(x) == 2 & op_cnt == 1 & rp_cnt == 1) original_plt_react()/residual_plt_react() else
            if(length(x) == 2 & tp_cnt == 1 & sp_cnt == 1) trend_plt_react()/seasonal_plt_react() else
              if(length(x) == 2 & tp_cnt == 1 & rp_cnt == 1) trend_plt_react()/residual_plt_react() else
                if(length(x) == 2 & sp_cnt == 1 & rp_cnt == 1) seasonal_plt_react()/residual_plt_react() else
      if(length(x) == 1 & op_cnt == 1) original_plt_react() else
        if(length(x) == 1 & tp_cnt == 1) trend_plt_react() else
          if(length(x) == 1 & sp_cnt == 1) seasonal_plt_react() else
            if(length(x) == 1 & rp_cnt == 1) residual_plt_react() else
              print('Select one or more plot elements')
      
    }
    if (length(input$checkbox_season) > 0) {
      seasonPlot_checkbox_filter(input$checkbox_season)
    }
    
  })
  
  
  ### Model Design and Testing
  
  ## Train-Test Split
  
  # Train-Test Split
  user_def_split_react <- reactive({
    input <- input$slider_traintest
    input/100
  })
  train_react <- reactive({
    train_test_split(ts, split_perc = user_def_split_react(), out.train=T)
  })
  test_react <- reactive({
    train_test_split(ts, split_perc = user_def_split_react(), out.test=T)
  })
  split_perc_react <- reactive({
    round(length(train_react())/(length(train_react())+length(test_react())), 2) #variable (subject to value used in train_test_split())
  })
  
  output$traintest_plot1 <- renderPlot({
    par(mfrow=c(2,1))
    plot(train_react(), 
         main=glue("Training Set: {100*split_perc_react()}%"), xlab="", ylab="", sub=glue("n={length(train_react())}"),
         ylim=c(min(min(train_react()), min(test_react())),
                max(max(train_react()), max(test_react()))))
    plot(test_react(), 
         main=glue("Testing Set: {100*(1-split_perc_react())}%"), xlab="", ylab="", sub=glue("n={length(test_react())}"),
         ylim=c(min(min(train_react()), min(test_react())),
                max(max(train_react()), max(test_react()))))
  })
  
  
  # Forecast with Stepwise ARIMA
  output$traintest_plot2 <- renderPlot({
    if (input$switch_traintest) {
      show_modal_spinner()
      fit_arima <- auto.arima(train_react(), max.order = 20, 
                              max.p = 10, max.d = 3, max.q = 10,
                              max.P = 10, max.D = 3, max.Q = 10,
                              stepwise = TRUE, seasonal = TRUE, stationary = FALSE,
                              ic = "aic", trace = TRUE)
      remove_modal_spinner()
      
      # Predict next X days of sales 
      forecast <- forecast(fit_arima, h = length(test_react()), level = c(80, 95, 99))
      
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
      sub.title = glue("Forecast horizon: {length(test_react())} days \n
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
        theme(plot.subtitle = element_text(lineheight = 0.55)) 
    } else {
      y <- c(1,3,2,4,3,5,4,6,5,7)
      x <- c(1:10)
      plot(x,y, title("NOTE: Please select 'Forecast' above to render forecast"))
    }
        
  })

  
  ## Manually Fit (S)ARIMA Model
  
  fit_arima_react <- reactive({
    arima(train_react(), 
          order=c(input$ar_nonseason,
                  input$diff_nonseason,
                  input$ma_nonseason), 
          seasonal=list(order=c(input$ar_season,
                                input$diff_season,
                                input$ma_season),
                        period=input$period_season))
  })
  
  forecast_arima_react <- reactive({
    forecast(fit_arima_react(), h = length(test_react()), level = c(80, 95, 99))
  })
  
  # Check to see how tight the lags are within the residual CIs (ACF/PACF)
  #tsdisplay(residuals(fit_arima), lag.max=30, main='Seasonal Model Residuals')
  
  # Evaluation of Algorithm Performance
  output$eval_alg_arima <- renderPrint({
    eval_forecast(ts, forecast_arima_react(), test=test_react(), train=train_react(), 
                  console=T, return.eval_tbl=F, print.eval_tbl=F)
  })  
  # Test vs. Prediction 
  output$arima_plot1 <- renderPlot({
    fc_accuracy_print(test_react(), forecast_arima_react())
  })
  
  
  ## Manually Fit (T)BATS Model
  
  input_boxcox_tbats_react <- reactive({
    if (identical(input$boxcox_tbats, "TRUE"))
      TRUE
    else
      FALSE
  })
  input_trend_tbats_react <- reactive ({
    if (identical(input$trend_tbats, "TRUE")) 
      TRUE
    else if (identical(input$trend_tbats, "FALSE")) 
      FALSE
  })
  input_trendDP_tbats_react <- reactive({
    if (identical(input$trendDP_tbats, "TRUE"))
      TRUE
    else
      FALSE
  })
  input_armaErrors_tbats_react <- reactive ({
    if (identical(input$armaErrors_tbats, "TRUE")) 
      TRUE
    else if (identical(input$armaErrors_tbats, "FALSE")) 
      FALSE
  })
  
  fit_tbats_react <- reactive({
    if (input$switch_tbats) {
      show_modal_spinner()
      fit_tbats <- tbats(train_react())
      remove_modal_spinner()
    }
    else {
      fit_tbats <-  tbats(train_react(),
                          use.box.cox = input_boxcox_tbats_react(),
                          use.trend = input_trend_tbats_react(),
                          use.damped.trend = input_trendDP_tbats_react(),
                          use.arma.errors = input_armaErrors_tbats_react())
    }
    fit_tbats
  })
  
  forecast_tbats_react <- reactive({
    forecast(fit_tbats_react(), h = length(test_react()), level = c(80, 95, 99))
  })
  
  # Evaluation of Algorithm Performance
  output$eval_alg_tbats <- renderPrint({
    eval_forecast(ts, forecast_tbats_react(), test=test_react(), train=train_react(), 
                  console=T, return.eval_tbl=F, print.eval_tbl=F)
  })  
  # Test vs. Prediction 
  output$tbats_plot1 <- renderPlot({
    fc_accuracy_print(test_react(), forecast_tbats_react())
  })
  
  
  ### Model Performance
  
  ## Set Training Split 
  
  user_def_split_react2 <- reactive({
    input <- input$slider_traintest2
    input/100
  })
  train_react2 <- reactive({
    train_test_split(ts, split_perc = user_def_split_react2(), out.train=T)
  })
  test_react2 <- reactive({
    train_test_split(ts, split_perc = user_def_split_react2(), out.test=T)
  })
  split_perc_react2 <- reactive({
    round(length(train_react2())/(length(train_react2())+length(test_react2())), 2) #variable (subject to value used in train_test_split())
  })
  
  ## (S)ARIMA
  fit_arima_react2 <- reactive({
    arima(train_react2(), 
          order=c(input$ar_nonseason2,
                  input$diff_nonseason2,
                  input$ma_nonseason2), 
          seasonal=list(order=c(input$ar_season2,
                                input$diff_season2,
                                input$ma_season2),
                        period=input$period_season2))
  })
  
  forecast_arima_react2 <- reactive({
    forecast(fit_arima_react2(), h = length(test_react2()), level = c(80, 95, 99))
  })
  
  
  ## (T)BATS
  input_boxcox_tbats_react2 <- reactive({
    if (identical(input$boxcox_tbats2, "TRUE"))
      TRUE
    else
      FALSE
  })
  input_trend_tbats_react2 <- reactive ({
    if (identical(input$trend_tbats2, "TRUE")) 
      TRUE
    else if (identical(input$trend_tbats, "FALSE")) 
      FALSE
  })
  input_trendDP_tbats_react2 <- reactive({
    if (identical(input$trendDP_tbats2, "TRUE"))
      TRUE
    else
      FALSE
  })
  input_armaErrors_tbats_react2 <- reactive ({
    if (identical(input$armaErrors_tbats2, "TRUE")) 
      TRUE
    else if (identical(input$armaErrors_tbats2, "FALSE")) 
      FALSE
  })
  
  fit_tbats_react2 <- reactive({
    fit_tbats <-  tbats(train_react2(),
                        use.box.cox = input_boxcox_tbats_react2(),
                        use.trend = input_trend_tbats_react2(),
                        use.damped.trend = input_trendDP_tbats_react2(),
                        use.arma.errors = input_armaErrors_tbats_react2())
    fit_tbats
  })
  
  forecast_tbats_react2 <- reactive({
    forecast(fit_tbats_react2(), h = length(test_react2()), level = c(80, 95, 99))
  })
  
  ## Performance Plot
  
  # Plot TBATS
  output$tbats_plot2 <- renderPlot({
      plot_eval_forecast(ts, 
                         forecast_tbats_react2(), 
                         test_react2(), 
                         train_react2(), 
                         og_df.date_col = ts_df$date)
  })
  
  # Plot ARIMA
  output$arima_plot2 <- renderPlot({
      plot_eval_forecast(ts, 
                         forecast_arima_react2(), 
                         test_react2(), 
                         train_react2(), 
                         og_df.date_col = ts_df$date)
  })
  
 
  
  
  #slider_traintest2
  #switch_outliers2
  
}


####################################################################################
###### Create Shiny object / Run application
shinyApp(ui = ui, server = server)