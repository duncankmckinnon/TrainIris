#Duncan McKinnon
#Train NN

library(shiny)
library(plotly)
library(magrittr)
library(assertthat)
source('DeepNN.R')
source('ShallowNN.R')
source('LogisticReg.R')

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel('Testing Model Performance Using the Iris Dataset'),
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      sidebarPanel(
         radioButtons("learn_method", "Learning Method", 
                      choices = c('Logistic Regression',  'Neural Net', 'Deep NN'), selected = 'Neural Net'),
         sliderInput("train_size", "Training Set Size", min = 10, max = 150, value = 100, step = 1, round = T),
         sliderInput("alpha", "Learning Rate", min = 0.001, max = 0.999, value = 0.010, step = 0.001),
         sliderInput("num_iters", "Training Iterations", min = 5, max = 50, value = 15, step = 1),
         conditionalPanel(
           condition = "input.learn_method == 'Neural Net'",
           numericInput("n_h", "Nuerons", value = 5, min = 2, max = 20, step = 1)
          ),
         conditionalPanel(
           condition = "input.learn_method == 'Deep NN'",
           numericInput("n_h", "Nuerons", value = 5, min = 2, max = 20, step = 1),
           radioButtons("layers", "Hidden Layers", choices = c(1, 2, 3, 4, 5), selected = 3)
         ),
         actionButton("run_model", "Train Model")
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
          tabsetPanel(
           tabPanel("Performance Plot", plotlyOutput("performplot")),
           tabPanel("Cost Plot" , plotlyOutput(outputId = "costplot")),
           tabPanel("Perfomance Summary", tableOutput("performtable"))
           )
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  model_data <- eventReactive(input$run_model, 
  {
    if(input$learn_method == "Logistic Regression")
    {
       return(LR_Sample(train_size = input$train_size, alpha = input$alpha, num_iters = input$num_iters))
    }
    else if(input$learn_method == "Neural Net")
    {
       return(NN_Sample(train_size = input$train_size, n_h = input$n_h, alpha = input$alpha, num_iters = input$num_iters))
    } 
    else if(input$learn_method == "Deep NN")
    {
       return(Deep_NN_Sample(train_size = input$train_size, n_h = array(input$n_h, input$layers), alpha = input$alpha, num_iters = input$num_iters))
    }
  })
  
  
  output$costplot <- renderPlotly(
    {
      cost_data <- unlist(model_data()$Model['costs'])
      p <- plot_ly() %>%
           add_lines(name = "Cost over Training Iteration", x = 1:length(cost_data), y = cost_data)
      return(p)
    })
  
  output$performplot <- renderPlotly(
    {
      train_data <- unlist(model_data()$YTrain)
      result_train_data <- unlist(model_data()$Model['Train_Vals']) 
      test_data <- unlist(model_data()$YTest)
      result_test_data <- unlist(model_data()$Model['Test_Vals'])
      n <- plot_ly() %>%
        add_markers(name = "Training Values", x = 1:length(train_data), y = train_data, color = "Actual Training Values") %>%
        add_markers(name = "Model Values", x = 1:length(result_train_data), y = result_train_data, color = "Model Training Results")
      p <- plot_ly() %>%
        add_markers(name = "Testing Values", x = 1:length(test_data), y = test_data, color = "Actual Testing Values") %>%
        add_markers(name = "Model Values", x = 1:length(result_test_data), y = result_test_data, color = "Model Testing Results")
      return(subplot(n, p))
    }
  )
  
  output$performtable <- renderTable(
    {
      
      return(data.frame('Training Accuracy' = unlist(model_data()$Model['Train_Per']), 'Testing Accuracy' = unlist(model_data()$Model['Test_Per'])))
      #model_data()$Model['w']
      #model_data()$Model['b']
    }
  )
}

# Run the application 
shinyApp(ui = ui, server = server)

