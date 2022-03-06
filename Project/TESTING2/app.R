# ORIGINAL PROGRAM PULLED FROM INTERNET



library(shiny)
library(shinyWidgets)
library(tidyverse)

ui <- fluidPage(
  
  selectInput("disp", "Disp", choices = unique(sort(mtcars$disp)), selected = 275.8),
  selectInput("hp", "hp", choices = unique(sort(mtcars$hp)), selected = 180),
  div(style="display:inline-block", textInput(('sample_name'), label = 'Sample Name',width = 200)),
  div(style="display:inline-block", actionButton(('select_sample'),icon = icon('save'), label = 'Save Sample')),
  panel(h2("User Created Inputs go here"),
        uiOutput("checkboxes")
  ),
  DT::dataTableOutput("cardata")
)

server <- function(input,output,session) {
  
  rv <- reactiveValues(filters = data.frame(
    id = character(),
    disp = double(),
    hp = double()
  ))
  
  compileData <- reactive({
    if (is.null(input$checkboxes)) {
      mtcars %>% filter(hp == input$hp & disp == input$disp)
    } else {
      merge(mtcars, rv$filters[rv$filters$id %in% input$checkboxes, ], by = c("disp", "hp"))
    }
  })
  
  output$cardata <- DT::renderDataTable({
    compileData()
  })
  
  observeEvent(input$select_sample, ignoreInit = FALSE, {
    rv$filters <- rbind(rv$filters, data.frame(
      id = input$sample_name,
      disp = input$disp,
      hp = input$hp
    )
    )
  })
  
  output$checkboxes <- renderUI({
    checkboxGroupInput("checkboxes", "Filters:", choices = rv$filters$id, selected = NULL)
  })
}

shinyApp(ui,server)