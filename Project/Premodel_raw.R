# For outlier detection
library(e1071)
library(kernlab)

# For stationarity test and ts decomposition
library(tseries)

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
  all_series <- data.frame(ts_fun, rolling_avg, lower_bound, upper_bound)
  
  p <- ggplot(data = all_series, aes(x = date)) +
    geom_line(aes(y = sales, color = "turquoise4")) +
    theme_minimal() +
    labs(x="", y="Sales ($)", title="Total Daily Sales (2020)") +
    theme(plot.title = element_text(hjust=0.5, size=20, face="bold")) +
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
ggplot(ts, aes(sales)) + 
  geom_histogram(color="turquoise4", fill="lightgreen",
                 alpha=.5, bins=100) +
  geom_vline(xintercept = 10000, color="red", linetype="dashed") +
  labs(x="Sales ($)", y="Frequency", title="Daily Sales Distribution")

ggplot(ts, aes(sales)) + 
  geom_density(color="turquoise4", fill="lightgreen",
               alpha=.4) +
  geom_vline(xintercept = 10000, color="red", linetype="dashed") +
  labs(x="Sales ($)", y="Density", title="Daily Sales Density Function")

# Plot a simple boxplot
ggplot(ts, aes(sales)) +
  geom_boxplot(color="turquoise4", fill="lightgreen",
               alpha=.4, outlier.shape = 1, outlier.color = "red") +
  geom_vline(xintercept = 10000, color = "red", linetype="dashed") +
  theme(axis.ticks = element_blank(), axis.text.y = element_blank()) +
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
                  color = "turquoise4")) +
    geom_point(aes(x = index, y = sales), 
               data = . %>% filter(outlier %in% 1), color = 'red') +
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

detect_outliers(ts, perc = .05)

# Interpolate outliers to remove them
remove_outliers <- function (ts, outliers_idx, return_df = TRUE, plot_ts = TRUE) {
  
  ts_clean <- ts
  ts_clean$sales[outliers_idx] <- NA
  ts_clean$sales <- na.approx(ts_clean$sales, method='linear') #interpolation
  
  # plot the two series
  ts_clean$og_sales <- ts$sales
  n_outliers <- ts_outliers %>% count(outlier)
  
  p <- ggplot(ts_clean) +
    geom_line(aes(x=date, y=og_sales, color="red")) +
    geom_line(aes(x=date, y=sales, color="turquoise4")) +
    labs(x="Date", y="Sales", title=glue("Outliers Removed: Found {n_outliers[2,2]}")) +
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
ts_outliers <- detect_outliers(ts, perc = .05)
# outliers' index position
outliers_index_pos <- ts_outliers[ts_outliers$outlier == 1, 3] #'3' is the index column we created
# exclude outliers
ts_clean <- remove_outliers(ts, outliers_idx = outliers_index_pos)


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
                 labels=c("Jan. 2013", "Jul", "Jan. 2014", "Jul", "Jan. 2015", "Jul"))
                 
  # pacf (for AR) and acf (for MA)
  lower_ylim <- min(min(acf(ts$sales, plot=F)$acf),
                    min(pacf(ts$sales, plot=F)$acf))   #automating graph alignment
  acf_plot <- ggAcf(ts$sales, lag.max=maxlag, ci=0.95) + ylim(c(lower_ylim, 1)) +
    labs(y="", title="Autocorrelation (for MA component)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold")) 
  pacf_plot <- ggPacf(ts$sales, lag.max=maxlag, ci=0.95) + ylim(c(lower_ylim, 1)) +
    labs(y="", title="Partial Autocorrelation (for AR component)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5, size=12, face="bold"))
  
  p / (acf_plot + pacf_plot) #arrange plots with patchwork::
  
}

plot_stationarity_test(ts, sample = 0.20, maxlag=30)

# Stabilize the mean by differences the ts
lag_ts <- ts %>% mutate_all(lag, n=1)
lag_ts$sales <- ts$sales - lag_ts$sales
lag_ts$date <- ts$date
lag_ts <- lag_ts[rowSums(is.na(lag_ts))==0,]

plot_stationarity_test(lag_ts, sample = 0.20, maxlag=30)


## Seasonality Analysis
units <- ts(ts$sales, frequency = 7)      #weekly seasonality
decomp <- stl(units, s.window='periodic')      
seasonal <- decomp$time.series[,1]
seasonal <- cbind(data.frame(seasonal), ts$date)
colnames(seasonal) <- c("seasonal", "date")
trend <- decomp$time.series[,2]
trend <- cbind(data.frame(trend), ts$date)
colnames(trend) <- c("trend", "date")
residual <- decomp$time.series[,3]
residual <- cbind(data.frame(residual), ts$date)
colnames(residual) <- c("residual", "date")

original_plt <- ggplot(ts, aes(x=date, y=sales)) +
  geom_line(color="#006699") +
  theme_minimal() +
  labs(title="Original series") +
  theme(axis.text.x = element_blank(), axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
        plot.margin = unit(c(0,0,0,0), "cm"),
        panel.border = element_rect(color="black", fill=NA, size=.5)) +
  scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                  "2014-01-01", "2014-05-01", "2014-09-01",
                                  "2015-01-01", "2015-05-01", "2015-09-01")))

trend_plt <- ggplot(trend, aes(x=date, y=trend)) +
  geom_line(color="#006699") +
  theme_minimal() +
  labs(title="Trend") +
  theme(axis.text.x = element_blank(), axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
        plot.margin = unit(c(0,0,0,0), "cm"),
        panel.border = element_rect(color="black", fill=NA, size=.5)) +
  scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                  "2014-01-01", "2014-05-01", "2014-09-01",
                                  "2015-01-01", "2015-05-01", "2015-09-01")))

seasonal_plt <- ggplot(seasonal, aes(x=date, y=seasonal)) +
  geom_line(color="#006699") +
  theme_minimal() +
  labs(title="Seasonal") +
  theme(axis.text.x = element_blank(), axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
        plot.margin = unit(c(0,0,0,0), "cm"),
        panel.border = element_rect(color="black", fill=NA, size=.5)) +
  scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                  "2014-01-01", "2014-05-01", "2014-09-01",
                                  "2015-01-01", "2015-05-01", "2015-09-01")))
  
residual_plt <- ggplot(residual, aes(x=date, y=residual)) +
  geom_line(color="#006699") +
  theme_minimal() +
  labs(title="Residual") +
  theme(axis.title.y = element_blank(),
        plot.title = element_text(hjust=0.5, vjust=-.5, size=10),
        plot.margin = unit(c(0,0,0,0), "cm"),
        panel.border = element_rect(color="black", fill=NA, size=.5)) +
  scale_x_date(breaks = as.Date(c("2013-01-01", "2013-05-01", "2013-09-01",
                                  "2014-01-01", "2014-05-01", "2014-09-01",
                                  "2015-01-01", "2015-05-01", "2015-09-01")),
               date_labels = "%Y-%m")   #unfortunately can't get date_breaks = "4 months" to pull correct labels
  

original_plt /
  trend_plt /
  seasonal_plt /
  residual_plt




