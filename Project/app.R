##### Load R packages

# For Shiny app
library(shiny)
library(shinythemes)
library(shinyWidgets)

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

##### Import Data
df <- read.csv("Data/sales_train.csv")

## Create "sales" series with a daily frequency datetime index
# Format datetime column
df$date <- as.Date(df$date, "%d.%m.%Y")

# Create time series
ts <- df %>%
  group_by(date) %>%
  summarise(sales = sum(item_cnt_day))


##### User-Defined Functions

## Function to create a trends plot with rolling statistics (mean, CI)
trendy_plot <- function (ts, plot_ma=TRUE, plot_intervals=TRUE, window=5) {
  
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
remove_outliers <- function (ts, outliers_idx, ts_outliers, return_df = TRUE, plot_ts = TRUE) {
  
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
                                 h4(p("Title 1")),
                                 h5(p("Subtitle"),
                                    p("something 1"),
                                    p("something 2")
                                 )
                          ),
                          column(6,
                                 h4(p("Title 2")),
                                 h5(p("Something 1")
                                 )
                          )
                        )
               ), # tabpanel
              tabPanel("Overview", fluid = TRUE,
                       fluidRow(
                         h3(p("Project Overview")),
                         h4(p("The Series"))
                       ),
                       fluidRow(
                         plotOutput("plt_ts")
                       ),
                       fluidRow(
                         column(6,
                                h4(p("Outline")),
                                h5(p("Step 1"),
                                   p("Step 2"),
                                   p("Step 3")
                                )
                         ),
                         column(6,
                                h4(p("Outline")),
                                h5(p("Step 1"),
                                   p("Step 2"),
                                   p("Step 3")
                                )
                         )
                       )
              ) # tabPanel
    ), # navbarMenu
    
    tabPanel("Navbar 1", fluid = TRUE, icon = icon("balance-scale-right"),
      sidebarPanel(
        tags$h3("Input:"),
        textInput("txt1", "Given Name:", ""),
        textInput("txt2", "Surname:", "")
      ), # sidebarPanel
      
      mainPanel(h1("Header 1"),
                h4("Output 1"),
                verbatimTextOutput("txtout")) # mainPanel
    ), # Navbar 1, tabPanel
    
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
                            helpText("Note: Algorithm can detect outliers at any threshold,
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
                          helpText(h2("NOTE TO SELF: the checkboxgroups would probably look better sitting above the plot vs. to the left."))
                        ),
                        mainPanel(
                          plotOutput("season_plot")
                        )
                        
               ) # tabPanel
               
    ), # Time Series Analysis, navbarMenu
    
    tabPanel("Model Design & Testing", fluid = TRUE, icon = icon('chart-bar'),
             "This panel is intentionally left blank"),
    tabPanel("My Analysis", fluid = TRUE, icon = icon('chart-line'),
             "This panel is intentionally left blank")
    
  ) # navbarPage
  
) # fluidPage


####################################################################################
##### Define server function
server <- function(input, output) {
  
  ### Intro ()
  
  # basic ts plot
  output$plt_ts <- renderPlot({
    ggplot(ts, aes(x=date, y=sales)) +
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
    trendy_plot(ts, 
                window = input$slider_trendanalysis,
                plot_ma = (input$radio_trendanalysis %in% c("Rolling Average", "Both")),
                plot_intervals = (input$radio_trendanalysis %in% c("Bollinger Bands", "Both")))
  })
  
  ## Outliers
  
  # Distribution Plots
  output$outlier_dist_plot <- renderPlot({
    if (input$radio_outliers1 == "Histogram") {
      # Plot a simple histogram & PDF/CDF
      ts %>%
        ggplot(aes(sales)) +
          geom_histogram(color="turquoise4", fill="lightgreen", alpha=.5, bins=100) +
          theme_standard +
          vline_outliers +
          labs(x="Sales ($)", y="Frequency", title="Daily Sales Distribution")
    } else if (input$radio_outliers1 == "Density Function") {
      # Plot a simple CDF/PDF
      options(scipen=10000)
      ts %>%
        ggplot(., aes(sales)) + 
        geom_density(color="turquoise4", fill="lightgreen", alpha=.4) +  
        theme_standard + 
        vline_outliers + 
        labs(x="Sales ($)", y="Density", title="Daily Sales Density Function")
    } else {
      # Plot a simple boxplot
      ts %>% 
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
        detect_outliers(ts, perc = slider_outliers_react())
      })
      # outliers' index position
      outliers_index_pos_react <- reactive({
        ts_outliers_react()[ts_outliers_react()$outlier == 1, 3] #'3' is the index column we created
      })
      # exclude outliers
      ts_clean <- remove_outliers(ts, ts_outliers = ts_outliers_react(), outliers_idx = outliers_index_pos_react())
    } else {
      detect_outliers(ts, return_df = FALSE,
                      gamma = 0.01,
                      perc = slider_outliers_react())
    }
  })
  
  ## Stationarity Test
  
  # Augmented Dickey-Fuller Test
  output$station_test_plot <- renderPlot({
    if (!input$switch_station) {
      plot_stationarity_test(ts, 
                             sample = input$slider_station1, 
                             maxlag = input$slider_station2)
    } else {
      # Stabilize the mean by differencing the ts
      lag_ts_react <- reactive({
        lag_ts_pre <- ts %>% 
          mutate_all(lag,n=1)
        lag_ts_pre$sales <- ts$sales - lag_ts_pre$sales
        lag_ts_pre$date <- ts$date
        lag_ts <- lag_ts_pre[rowSums(is.na(lag_ts_pre))==0,]
      })
      plot_stationarity_test(lag_ts_react(), 
                             sample = input$slider_station1,
                             maxlag = input$slider_station2)
    }
  })
  
  ## Seasonality/Decomposition
  
  ts$sales %.>% 
    { units <- ts(., frequency = 7) } %.>%   #weekly seasonality
    { decomp <- reactive({
        stl(., s.window='periodic')
    }) } 
  
  # Original Plot
  original_plt_react <- reactive({
    ggplot(ts, aes(x=date, y=sales)) +
      geom_line(color="#006699") +
      theme_4panel +
      labs(title="Original Series") +
      date_breaks_4panel
  })
  
  # Trend Plot
  trend_plt_react <- reactive({
    decomp() %.>%
      { .$time.series[,2] } %.>%
      { trend <- cbind(data.frame(.), ts$date) } %.>%
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
      { seasonal <- cbind(data.frame(.), ts$date) } %.>%
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
  
  
}


####################################################################################
###### Create Shiny object/ Run application
shinyApp(ui = ui, server = server)