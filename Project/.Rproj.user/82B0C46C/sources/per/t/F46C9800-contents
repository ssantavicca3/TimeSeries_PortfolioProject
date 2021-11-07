# Primary tutorial: https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205

# For outlier detection
library(e1071)
library(kernlab)

# For stationarity test and ts decomposition
library(tseries)       #adf test & ts()
library(urca)          #adf test for cv table

# For visualizations
library(ggplot2)
library(highcharter)   #htmlwidgets
library(dygraphs)      #htmlwidgets
library(patchwork)     #multi-panel ggplots
library(forecast)      #ggAcf and ggPacf plots

# For gen. purposes
library(dplyr)         #manipulation
library(magrittr)      #pipe & subsetting secondary series w/in ggplot
library(zoo)           #rolling stats and interpolation 
library(glue)          #string magic
library(wrapr)         #pipe ggplot layers


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

# Assign common ggplot elements to be subbed later on
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

# Import data
df <- read.csv("Data/sales_train.csv")

## Create "sales" series with a daily frequency datetime index
# Format datetime column
df$date <- as.Date(df$date, "%d.%m.%Y")

# Create time series
ts <- df %>%
  group_by(date) %>%
  summarise(sales = sum(item_cnt_day))

# View our ts so far
summary(ts)
p <- ggplot(ts, aes(x=date, y=sales)) +
  geom_line(color="turquoise4") +
  theme_minimal() +
  labs(x="", y="Sales ($)", title="Total Daily Sales (2020)") +
  theme(plot.title = element_text(hjust=0.5, size=20, face="bold"))
p


## Trend Analysis
# Write function to display rolling statistics in the plot
trendy_plot <- function (ts, plot_ma=TRUE, plot_intervals=TRUE, window=5) {
  
  rolling_avg <- zoo::rollmean(ts$sales, k=window, fill = list(NA,NULL,NA))
  rolling_std <- zoo::rollapply(ts$sales, width=window, sd, fill = list(NA,NULL,NA))
  
  lower_bound <- rolling_avg - (1.96*rolling_std)
  upper_bound <- rolling_avg + (1.96*rolling_std)
  
  ts_fun <- ts
  
  p <- ts_fun %.>%
    data.frame(ts_fun, rolling_avg, lower_bound, upper_bound) %.>% 
    ggplot(., aes(x = date)) %.>%
    theme_standard %.>%
    labels_standard %.>%
    scale_color_manual(name = "Sales", 
                       values = "turquoise4",
                       labels = "Actual Values")
  
  if (plot_ma) {
    p <- p + geom_line(aes(y = rolling_avg, color = 'red')) +
      scale_color_manual(name = "Sales", 
                         values = c("red", "turquoise4"),
                         labels = c("Rolling Avg.", "Actual Values"))
  }
  
  if (plot_intervals) {
    p <- p + geom_ribbon(aes(x=date, ymax=upper_bound, ymin=lower_bound),
                         fill="grey70", alpha=.4) 
  }
  
  print(p)
  
}

# Inspect the plot for trends
trendy_plot(ts, window = 30)
trendy_plot(ts, window = 365) #NB: can I restrict the input data to the domain of the CIs?


## Outlier Detection
# Plot a simple histogram & PDF/CDF
ts %.>%
  ggplot(., aes(sales)) %.>%
  geom_histogram(color="turquoise4", fill="lightgreen", alpha=.5, bins=100) %.>%
  theme_standard %.>% 
  vline_outliers %.>%
  labs(x="Sales ($)", y="Frequency", title="Daily Sales Distribution")

# Plot a simple CDF/PDF
ts %.>%
  ggplot(., aes(sales)) %.>%  
  geom_density(color="turquoise4", fill="lightgreen", alpha=.4) %.>% 
  theme_standard %.>%
  vline_outliers %.>% 
  labs(x="Sales ($)", y="Density", title="Daily Sales Density Function")

# Plot a simple boxplot
ts %.>% 
  ggplot(., aes(sales)) %.>% 
  geom_boxplot(color="turquoise4", fill="lightgreen",
               alpha=.4, outlier.shape = 1, outlier.color = "red") %.>% 
  vline_outliers %.>%
  theme_standard %.>% 
  theme(axis.ticks = element_blank(), axis.text.y = element_blank()) %.>% 
  labs(x="Sales ($)", y="", title="B&W Plot of Daily Sales")


# Write function to automatically detect outliers in a time series
detect_outliers <- function (ts, perc=0.01, gamma=0.01, return_df=TRUE, plot_ts=TRUE) {
  
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
    labs(x="", y="", title=glue("Outliers Detection: Found {n_outliers[2,2]}")) +
    scale_color_manual(name = "Sales", 
                       values = c("turquoise4", "red"),
                       labels = c("Actual Values", "Outliers")) 
  
  # conditional function output
  if (plot_ts) {
    print(p)
  }
  
  if (return_df) {
    return(ts_outliers)
  }
  
}

#ts_outliers %>% count(outlier)  #use to groupby count

detect_outliers(ts_df, perc = .05)

# Interpolate outliers to remove them
remove_outliers <- function (ts, outliers_idx, return_df = TRUE, plot_ts = TRUE) {
  
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

# Detect outliers
ts_outliers <- detect_outliers(ts_df, perc = .05)
# outliers' index position
outliers_index_pos <- ts_outliers[ts_outliers$outlier == 1, 3] #'3' is the index column we created
# exclude outliers
ts_clean <- remove_outliers(ts_df, outliers_idx = outliers_index_pos)


## Stationarity Test
# Write a function to visualize stats and partial/autocorrelation and run ADF test
plot_stationarity_test <- function (ts, sample=0.20, maxlag=30) {
  
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
    annotate("text", x=as.Date("2015-07-01"), y=c(12000, 11250, 10500), 
             label=c(glue("95% CV: {cv_95}"), 
                     glue("99% CV: {cv_99}"),
                     glue("ADF Stat: {round(adf,2)}")))
  
                 
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
plot_stationarity_test(ts, sample = 0.20, maxlag=30)

# Stabilize the mean by differences the ts
lag_ts <- ts %>% mutate_all(lag, n=1)
lag_ts$sales <- ts$sales - lag_ts$sales
lag_ts$date <- ts$date
lag_ts <- lag_ts[rowSums(is.na(lag_ts))==0,]

plot_stationarity_test(lag_ts, sample = 0.20, maxlag=30)


## Seasonality Analysis
ts$sales %.>% 
  { units <- ts(., frequency = 7) } %.>%   #weekly seasonality
  { decomp <- stl(., s.window='periodic') } 

original_plt <- ggplot(ts, aes(x=date, y=sales)) +
  geom_line(color="#006699") +
  theme_4panel +
  labs(title="Original series") +
  date_breaks_4panel

trend_plt <- decomp %.>%
  { .$time.series[,2] } %.>%
  { trend <- cbind(data.frame(.), ts$date) } %.>%
  { colnames(trend) <- c("trend", "date") } %.>%
  ggplot(trend, aes(x=date, y=trend)) %.>%
    geom_line(color="#006699") %.>%
    theme_4panel %.>%
    labs(title="Trend") %.>%
    date_breaks_4panel

seasonal_plt <- decomp %.>%
  { .$time.series[,1] } %.>%
  { seasonal <- cbind(data.frame(.), ts$date) } %.>%
  { colnames(seasonal) <- c("seasonal", "date") } %.>%
  ggplot(seasonal, aes(x=date, y=seasonal)) %.>%
    geom_line(color="#006699") %.>%
    theme_4panel %.>%
    labs(title="Seasonal") %.>%
    date_breaks_4panel

residual_plt <- decomp %.>%
  { .$time.series[,3] } %.>%
  { residual <- cbind(data.frame(.), ts$date) } %.>%
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

original_plt /
  trend_plt /
  seasonal_plt /
  residual_plt