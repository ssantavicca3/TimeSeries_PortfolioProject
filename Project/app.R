##### Load R packages
library(shiny)
library(shinythemes)

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

# Create "%notin%" operator by negating "%in%"
`%notin%` <- Negate(`%in%`)


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

# Function to create a trends plot with rolling statistics (mean, CI)
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
                                      label = h3("Window length (k = days)"), 
                                      min = 1, max = 365, 
                                      value = 30),
                          checkboxGroupInput(inputId = "checkGroup", 
                                             label = h3("Checkbox group"),
                                             choices = c("Rolling Average",
                                                         "Bollinger Bands"))
                        ),
                        mainPanel(
                          plotOutput("trends_plot")
                        )
                          
               ), # tabpanel
               
               tabPanel("Outliers", fluid = TRUE,
                        titlePanel("Outlier Detection"),
                    
               ), # tabPanel
               
               tabPanel("Stationarity", fluid = TRUE,
                        titlePanel("Stationarity Test"),
                        
               ), # tabpanel
               tabPanel("Seasonality", fluid = TRUE,
                        titlePanel("Time Series Decomposition"),
                        
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
  
  # Slider - window length
  # checkGroup <- reactive({
  #   # if (input$checkGroup %notin% c("RA", "BB")) {
  #   #   c(FALSE, FALSE)
  #   # } else 
  #   if (input$checkGroup %notin% "RA" & input$checkGroup %in% "BB") {
  #     c(FALSE, TRUE)
  #   } else if (input$checkGroup %in% "RA" & input$checkGroup %notin% "BB") {
  #     c(TRUE, FALSE)
  #   } else {
  #     c(TRUE, TRUE)
  #   }
  # })
  output$trends_plot <- renderPlot({
      trendy_plot(ts, 
                  window = input$slider_trendanalysis,
                  plot_ma = (input$checkGroup == "Rolling Average"),
                  plot_intervals = (input$checkGroup == "Bollinger Bands"))
  })
  
  
}


####################################################################################
###### Create Shiny object/ Run application
shinyApp(ui = ui, server = server)