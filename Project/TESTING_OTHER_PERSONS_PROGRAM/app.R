library(shiny)
library(shinyWidgets)
library(tidyverse)
library(zoo)
library(forecast)
library(glue)
library(wrapr)      
library(data.table) 


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


##### Import Data
df <- read.csv("C:/Users/ssantavicca3/OneDrive/Projects/TimeSeries_PortfolioProject/Project/Data/sales_train.csv")

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


# Function to evaluate algorithm performance
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


## Function to fit a model (stick this in the next function or make way to merge with eval_forecast())
fc_fn <- function (ts=ts, split_perc=NULL, train = train, test = test,
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
                   manual.tbats.spec = tbats(y = train,
                                             use.box.cox = NULL,
                                             use.trend = NULL,
                                             use.damped.trend = NULL,
                                             use.arma.errors = NULL,
                                             trace = TRUE)) { 
  
  # create the training and test sets
  if (is.null(split_perc)) {
    train <- train
    test <- test
    split_perc = train/(train+test)
  } else {
    train <- train_test_split(ts, split_perc = split_perc, out.train = T)
    test <- train_test_split(ts, split_perc = split_perc, out.test = T)
  }
 
  
  
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

#########################################################################################################################
ui <- fluidPage(
  navbarPage(
    "Time Series ML Forecasting",
    navbarMenu("Model Design & Testing",
               tabPanel("Train-Test Split", fluid = T,
                        fluidRow(
                          column(4,
                                 sliderInput(inputId = "slider_traintest",
                                             label = h3("Training set (%)"),
                                             min = 1, max = 100,
                                             value = 85)
                          )
                        ),
                        titlePanel("Sample Split"),
                        plotOutput("traintest_plot1")
                 
               ),
               
               tabPanel("ARIMA", fluid = T,
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
                            hr(),
                            #Put option to save model here
                            div(
                              style="display:inline-block",
                              textInput((inputId = 'name_arima_model'),
                                        label = 'Name your model',
                                        width = 200)
                            ),
                            div(
                              style="display:inline-block",
                              actionButton((inputId = 'select_arima_model'),
                                           icon = icon('save'),
                                           label = 'Save Model')
                            )
                          ),
                          mainPanel(
                            #Put the "accuracy of algorithm" & "test vs. prediction" plots here
                            verbatimTextOutput("eval_alg_arima")
                          )
                        )
               ),
    ),           
               
    tabPanel("Model Comparisons",
             panel(h2("User Created Inputs go here"),
                   uiOutput("checkboxes")),
             hr(),
             plotOutput("eval_comp_table")
            )
  )
)


########################################################################################################################
server <- function(input,output,session) {
  
  
  ## Train-Test Split
  
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

  
  ##############################################     #############################################     ######################################
  
  # storing ARIMA parameter choices in a list/df
  rv_arima <- reactiveValues(params = data.frame(
    id = character(),
    p = double(),
    d = double(),
    q = double(),
    P = double(),
    D = double(),
    Q = double(),
    m = double()
  ))
  
  eval_comp_tbl <- reactive({
    tibble(
      "Measures" = c(
        "Residuals mean",
        "Residuals sd",
        "Error mean",
        "Error sd",
        "MAE",
        "MAPE (%)",
        "MSE",
        "RMSE",
        "Test mean",
        "RMSE/Test mean (%)"
      )
    )
  })
    
  
  observeEvent(input$select_arima_model, ignoreInit=F, {
    rv_arima$params <- rbind(rv_arima$params, data.frame(
      id = input$name_arima_model,
      p = input$ar_nonseason,
      d = input$diff_nonseason,
      q = input$ma_nonseason,
      P = input$ar_season,
      D = input$diff_season,
      Q = input$ma_season,
      m = input$period_season
    )
    )
    
    eval_comp_tbl_i <- fc_fn(
      ts=ts,
      train=train_react(),
      test=test_react(),
      modelvar = "arima",
      eval_fc_output = "return eval object",
      manual.arima.spec = arima(
        train_react(), 
        order=c(input$ar_nonseason,
                input$diff_nonseason,
                input$ma_nonseason), 
        seasonal=list(order=c(input$ar_season,
                              input$diff_season,
                              input$ma_season),
                      period=input$period_season))
    )
    
    eval_comp_tbl() <- tibble(
      eval_comp_tbl(),
      eval_comp_tbl_i[2]
    )
    
    
    
  })
  
  # render the user-saved models as checkboxes
  output$checkboxes <- renderUI({
    checkboxGroupInput("checkboxes", "Filters:", choices = rv_arima$params$id, selected = NULL)
  })
  
  # render the eval_table 
  output$eval_comp_table <- renderPlot({
    ggplot() + geom_table_npc(data=eval_comp_tbl, label=list(eval_comp_tbl), 
                              npcx=0.5, npcy=0.5, size=4, 
                              table.theme=ttheme_gtstripes) + theme_minimal() +
      theme(plot.title = element_text(hjust=0.5, vjust=2, size=11))
  })
  
  
  #
  
  #________________________________#
  # ui <- fluidPage(
  # 
  #   selectInput("disp", "Disp", choices = unique(sort(mtcars$disp)), selected = 275.8),
  #   selectInput("hp", "hp", choices = unique(sort(mtcars$hp)), selected = 180),
  #   div(style="display:inline-block", textInput(('sample_name'), label = 'Sample Name',width = 200)),
  #   div(style="display:inline-block", actionButton(('select_sample'),icon = icon('save'), label = 'Save Sample')),
  #   panel(h2("User Created Inputs go here"),
  #         uiOutput("checkboxes")
  #   ),
  #   DT::dataTableOutput("cardata")
  # )
  # 
  # server <- function(input,output,session) {
  # 
  #   rv <- reactiveValues(filters = data.frame(
  #     id = character(),
  #     disp = double(),
  #     hp = double()
  #   ))
  # 
  #   observeEvent(input$select_sample, ignoreInit = FALSE, {
  #     rv$filters <- rbind(rv$filters, data.frame(
  #       id = input$sample_name,
  #       disp = input$disp,
  #       hp = input$hp
  #     )
  #     )
  #   })
  # 
  #   output$checkboxes <- renderUI({
  #     checkboxGroupInput("checkboxes", "Filters:", choices = rv$filters$id, selected = NULL)
  #   })
  #   
  #   compileData <- reactive({
  #     if (is.null(input$checkboxes)) {
  #       mtcars %>% filter(hp == input$hp & disp == input$disp)
  #     } else {
  #       merge(mtcars, rv$filters[rv$filters$id %in% input$checkboxes, ], by = c("disp", "hp"))
  #     }
  #   })
  #   
  #   output$cardata <- DT::renderDataTable({
  #     compileData()
  #   })
  #   
  # }
  #___________________________________#
  
  
  
  
   

}

shinyApp(ui,server)