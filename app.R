#Duncan McKinnon
#Train NN

library(shiny)
library(plotly)
library(RColorBrewer)
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
        # radioButtons("data_method", "Input or Built-in dataset", choices = c("built-in", "csv data input")),
         radioButtons("learn_method", "Learning Method", 
                      choices = c('Logistic Regression',  'Neural Net', 'Deep NN'), selected = 'Neural Net'),
         sliderInput("train_size", "Training Set Size", min = 10, max = 150, value = 100, step = 1, round = T),
         sliderInput("alpha", "Learning Rate", min = 0.001, max = 0.999, value = 0.010, step = 0.001),
         sliderInput("num_iters", "Training Iterations", min = 1, max = 1000, value = 15, step = 1),
        # conditionalPanel(
        #   condition = "input.data_method == 'csv data input'",
        #   fileInput("datafile", "Choose CSV file", accept = c(".csv"))
        # ),
        conditionalPanel(
           condition = "input.learn_method == 'Logistic Regression'",
            h4("Logistic regression differentiates between items in and out of the set"),
            checkboxGroupInput("iris_type", label = "Iris Type(s)", choices = c("setosa", "virginica", "versicolor"), selected = "virginica")
         ),
         conditionalPanel(
           condition = "input.learn_method == 'Neural Net'",
           h4("Neural network trains through one layer of neurons using an activation function"),
           numericInput("n_hNN", "Nuerons", value = 5, min = 2, max = 20, step = 1)
          ),
         conditionalPanel(
           condition = "input.learn_method == 'Deep NN'",
           h4("Deep neural network trains through several layers of neurons with the same activation functions"),
           numericInput("n_hDNN", "Nuerons", value = 5, min = 2, max = 20, step = 1),
           radioButtons("layers", "Hidden Layers", choices = c(1, 2, 3, 4, 5), selected = 3),
           checkboxInput("use_momentum", "with Momentum", value = F)
         ),
         h4("Display categorized training result or raw training estimates"),
         checkboxInput("rawOut", "Categorized/Raw", value = F),
         actionButton("run_model", "Train Model")
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
          tabsetPanel(
           tabPanel("Performance Plot", 
                    plotlyOutput(outputId = "performplot")),
           tabPanel("Cost Plot" , 
                    plotlyOutput(outputId = "costplot")),
           tabPanel("Perfomance Summary", 
                    tableOutput("performtable")),
           tabPanel("Datasets and Results", 
                    h4("Training Dataset and Result"),
                    dataTableOutput("TrainingData"),
                    h4("Testing Dataset and Result"),
                    dataTableOutput("TestingData")),
           tabPanel("Data Representation",
                    plotlyOutput(outputId = "scatterMatrix"))
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
        return(LR_Sample(train_size = input$train_size, alpha = input$alpha, num_iters = input$num_iters, set = input$iris_type, raw = input$rawOut))
      }
      else if(input$learn_method == "Neural Net")
      {
        return(NN_Sample(train_size = input$train_size, n_h = input$n_hNN, alpha = input$alpha, num_iters = input$num_iters, act = "relu", raw = input$rawOut))
      } 
      else if(input$learn_method == "Deep NN")
      { 
        return(Deep_NN_Sample(train_size = input$train_size, n_h = array(input$n_hDNN, input$layers), alpha = input$alpha, num_iters = input$num_iters, act = "relu", raw = input$rawOut, momentum = input$use_momentum))
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
    })
  
    output$performtable <- renderTable(
    {
      return(data.frame('Training Accuracy' = unlist(model_data()$Model['Train_Per']), 'Training Correlation' = unlist(model_data()$Model['Train_Cor']), 'Testing Accuracy' = unlist(model_data()$Model['Test_Per']), 'Testing Correlation' = unlist(model_data()$Model['Test_Cor'])))
    })
    
    output$TrainingData <- renderDataTable(
    {
      return(cbind.data.frame(model_data()$XTrain, "Iris_Type" = as.vector(model_data()$YTrain), "Model_Result" = as.matrix(unlist(model_data()$Model['Train_Vals']), ncol = 1)))
    }, options = list(pageLength = 10))
  
    output$TestingData <- renderDataTable(
    {
      return(cbind.data.frame(model_data()$XTest, "Iris_Type" = as.vector(model_data()$YTest), "Model_Result" = as.matrix(unlist(model_data()$Model['Test_Vals']), ncol = 1)))
    }, options = list(pageLength = 10))
    
    output$scatterMatrix <- renderPlotly(
      {
        species <- c("Setosa", "Versicolor", "Virginica")
        n <- plot_ly(model_data()$XTrain) %>%
             add_markers(x = ~Petal.Width, y = ~Petal.Length, symbol = model_data()$YTrain, symbols = c("circle","cross","diamond"),
                      colors = "Spectral", color = abs(as.vector(unlist(model_data()$Model['Train_Vals'])) - model_data()$YTrain),
                      text = ~paste("Species = ", species[model_data()$YTrain],
                                    "</br>Sepal Length = ", Sepal.Length,
                                    "</br>Petal Length = ", Petal.Length,
                                    "</br>Sepal Width = ", Sepal.Width,
                                    "</br>Petal Width = ", Petal.Width)) %>%
                      layout(title = "Error By Flower", xaxis = list("Petal Width"), yaxis = list("Petal Length"))
        p <- plot_ly(model_data()$XTest) %>%
          add_markers(x = ~Petal.Width, y = ~Petal.Length, symbol = model_data()$YTest, symbols = c("circle","cross","diamond"), 
                      colors = "Spectral", color = abs(as.vector(unlist(model_data()$Model['Test_Vals'])) - model_data()$YTest),
                      text = ~paste("Species = ", species[model_data()$YTest],
                                   "</br>Sepal Length = ", Sepal.Length,
                                   "</br>Petal Length = ", Petal.Length,
                                   "</br>Sepal Width = ", Sepal.Width,
                                   "</br>Petal Width = ", Petal.Width)) %>%
                      layout(title = "Error By Flower", xaxis = list("Petal Width"), yaxis = list("Petal Length"))
        return(subplot(n, p))
      }
    )
}

# Run the application 
shinyApp(ui = ui, server = server)

