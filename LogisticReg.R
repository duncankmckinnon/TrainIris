#Logistic Regression Function
#Duncan McKinnon

source('matrixBroadcasting.R')
source('parseData.R')
source('activation.R')

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
  accuracy_Train <- 1 - sum(((YTrain) - pred_Train) ^ 2) / length(YTrain)
  cor_Train <- cor.test(YTrain, pred_Train)$estimate
  
  LRMod <- list("w" = vals$w, "b" = vals$b, "costs" = vals$costs, "is_diff" = raw, "Train_Per" = accuracy_Train, "Train_Cor" = cor_Train, "Train_Vals" = pred_Train)
  
#run system against testing data
  if(!is.null(XTest) && !is.null(YTest))
  {
    XTest <- as.matrix(XTest)
    YTest <- as.matrix(YTest)  
    pred_Test <- as.matrix(LRMod_predict(vals$w, vals$b, XTest, raw), nrow = 1)
    accuracy_Test <- 1 - sum(((YTest) - pred_Test) ^ 2) / length(YTest)
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
  accuracy_Test <- 1 - sum(((YTest) - pred_Test) ^ 2) / length(YTest)
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


#Generate a sample model trained to differentiate between flowers in the iris sample set.
#type = c("setosa", "versicolor", "virginica")
LR_Sample <- function(data_set = iris, xcol = 1:4, ycol = 5, train_size = 100, test_size = 50, alpha = 0.01, num_iters = 10, raw = F, set = "setosa", type = "", regularize = F)
{
  
  
  if(train_size > dim(data_set)[1])
  {
    train_size <- dim(data_set)[1] - 10
  }
  else if(train_size < 20)
  {
    train_size <- 20
  }
  
  dataset <- parseModelData(data_set, x_cols = xcol, y_cols = ycol, train_size = train_size, test_size = test_size)
  if(typeof(set) == "closure")
  {
    dataset$YTrain <- as.numeric(set(dataset$YTrain))
    dataset$YTest <- as.numeric(set(dataset$YTest))
  }
  else
  {
    dataset$YTrain <- as.numeric(dataset$YTrain %in% set)
    dataset$YTest <- as.numeric(dataset$YTest %in% set)
  }
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
