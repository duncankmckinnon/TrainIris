#Logistic Regression Function
#Duncan McKinnon


LogisticRegression_Model <- function(XTrain, YTrain, alpha = 0.01, num_iters = 10, raw = F,  XTest = NULL, YTest = NULL)
{
#internal model function to perform gradient descent optimization on weights and offset
  optimize <- function(w, b, XTrain, YTrain, alpha, num_iters)
  {
    costs <- c()
    for(i in 1:num_iters)
    {
      vals <- propogate(w, b, XTrain, YTrain)
      
      w <- w - (alpha * vals$dw)
      b <- b - (alpha * vals$db)
      costs <- c(costs, vals$cost)
    }
    return(list("w" = w, "b" = b, "dw" = vals$dw, "db" = vals$db,  "costs" = costs))
  }
  
#internal model function to perform forward propogation to get estimates based on current weights and offset
# and back propogation for next optimization step
  propogate <- function(w, b, XTrain, YTrain)
  {
    m <- dim(XTrain)[2]
    
    guess <- activation(XTrain %*% w + b, type = "sigmoid")
    
    cost <- (-1/m) * sum((t(YTrain) %*% log(guess)) + (1 - t(YTrain)) %*% log(1 - guess))
    
    dw <- (1/m) * t(XTrain) %*% (guess - YTrain)
    
    db <- (1/m) * sum(guess - YTrain)
    
    return(list("dw" = dw, "db" = db, "cost" = cost))
  }
  
#initialization of variables for training phase
  XTrain <- as.matrix(XTrain)
  YTrain <- as.matrix(YTrain)
  
  w = matrix(0.5, nrow = dim(XTrain)[2])
  b = 0
  
#train system on training data
  vals <- optimize(w, b, XTrain, YTrain, alpha, num_iters)
  
#run system against training results
  pred_Train <- as.matrix(LRMod_predict(vals$w, vals$b, XTrain, raw), nrow = 1)
  accuracy_Train <- 1 - sum(abs((YTrain) - pred_Train)) / length(YTrain)
  cor_Train <- cor.test(YTrain, pred_Train)$estimate
  
  LRMod <- list("w" = vals$w, "b" = vals$b, "costs" = vals$costs, "is_diff" = raw, "Train_Per" = accuracy_Train, "Train_Cor" = cor_Train, "Train_Vals" = pred_Train)
  
#run system against testing data
  if(!is.null(XTest) && !is.null(YTest))
  {
    XTest <- as.matrix(XTest)
    YTest <- as.matrix(YTest)  
    pred_Test <- as.matrix(LRMod_predict(vals$w, vals$b, XTest, raw), nrow = 1)
    accuracy_Test <- 1 - sum(abs((YTest) - pred_Test)) / length(YTest)
    cor_Test <- cor.test(YTest, pred_Test)$estimate
    LRMod[["Test_Per"]] = accuracy_Test
    LRMod[["Test_Cor"]] = cor_Test
    LRMod[["Test_Vals"]] = pred_Test
  }
  
  return(LRMod)
}

#Run existing model against a new dataset
Predict <- function(LRMod, XTest, YTest, raw = F)
{
  pred_Test <- as.matrix(LRMod_predict(LRMod$w, LRMod$b, XTest, raw), nrow = 1)
  accuracy_Test <- 1 - sum(abs((YTest) - pred_Test)) / length(YTest)
  cor_Test <- cor.test(YTest, pred_Test)$estimate
  predModel <- list("values" = pred_Test, "Accuracy" = accuracy_Test, "Correlation" = cor_Test)
  return(predModel)
}

#Get prediction results for a set of parameters and data
LRMod_predict <- function(w, b, XTest, raw = F)
{
  if(!raw)
  {
    return(ifelse(activation(XTest %*% w + b, type = "sigmoid") > 0.5, 1, 0))
  }
  return(activation(XTest %*% w + b, type = "sigmoid"))
}

#Non-linear activation functions for determining classifications based on input
activation <- dget('activation.R')

#parse a dataset in a data frame into a training and sample set
parseModelData <- dget('parseData.R')

#Generate a sample model trained to differentiate between flowers in the iris sample set.
#type = c("setosa", "versicolor", "virginica")
LR_Sample <- function(train_size = 100, alpha = 0.01, num_iters = 10, raw = F, set = "versicolor", type = "")
{
  if(train_size > 140)
  {
    train_size <- 140
  }
  else if(train_size < 40)
  {
    train_size <- 40
  }
  
  dataset <- parseModelData(data_set = iris, x_cols = 1:4, y_cols = 5, train_size = train_size)
  dataset$YTrain <- as.numeric(dataset$YTrain %in% set)
  dataset$YTest <- as.numeric(dataset$YTest %in% set)
  LRMod <- LogisticRegression_Model(XTrain = dataset$XTrain, YTrain = dataset$YTrain, XTest = dataset$XTest, YTest = dataset$YTest, alpha = alpha, num_iters = num_iters, raw = raw)
  
  if(type == "")
  {
    return(list("XTrain" = dataset$XTrain, "YTrain" = dataset$YTrain, "XTest" = dataset$XTest, "YTest" = dataset$YTest, "Model" = LRMod))
  }
  else
  {
    return(LRMod[[type]])
  }
}
