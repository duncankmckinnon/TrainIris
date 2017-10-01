#Neural Network Model (1 hidden layer of n_h nodes)
#Duncan McKinnonx


NeuralNetwork_Model <- function(XTrain, YTrain, n_h = 4, alpha = 0.01, num_iters = 10, type = "tanH", XTest = NULL, YTest = NULL)
{
#internal model function to perform gradient descent optimization on weights and offset
  NN_optimize <- function(w, b, XTrain, YTrain, alpha, num_iters, type)
  {
    costs <- c()
    for(i in 1:num_iters)
    {
      vals <- NN_propagate(w, b, XTrain, YTrain, type)
      w[[1]] = w[[1]] - (alpha * vals$dw[[1]])
      b[[1]] = b[[1]] - (alpha * vals$db[[1]])
      w[[2]] = w[[2]] - (alpha * vals$dw[[2]])
      b[[2]] = b[[2]] - (alpha * vals$db[[2]])
      
      costs <- c(costs, vals$cost)
    }
    return(list("w" = w, "b" = b, "dw" = vals$dw,  "db" = vals$db,  "costs" = costs))
  }

#internal model function to perform forward propagation to get estimates based on current weights and offset
# and back propogation for next optimization step  
  NN_propagate <- function(w, b, XTrain, YTrain, type)
  {
    m <- dim(XTrain)[2]
    
    z1 <- (w[[1]] %*% XTrain) %+% b[[1]]
    
    a1 <- activation(z1, type)
    
    z2 <- (w[[2]] %*% a1) %+% b[[2]]
    
    a2 <- activation(z2, type)
    
    cost <- -(1/m) * sum((YTrain - t(a2))^2)
    
    dz2 <- a2 - t(YTrain)
    
    dw2 <- (1/m) * dz2 %*% t(a1)
    
    db2 <- (1/m) * colSums(t(dz2))
    
    dz1 <- (t(w[[2]]) %*% dz2) * activation(z1, type, T)
    
    dw1 <- (1/m) * dz1 %*% t(XTrain)
    
    db1 <- (1/m) * colSums(t(dz1))
    
    return(list("dw" = list(dw1, dw2), "db" = list(db1, db2), "cost" = cost))
  }
  
#initialization of variables for training phase
  XTrain <- t(as.matrix(XTrain))
  YTrain <- as.matrix(YTrain)

 #number of inner layer dimensions  
  n <- c(dim(XTrain)[1], n_h, dim(YTrain)[2])
  
  
  w <- list()
  b <- list()
  
  #initialize 2 levels of weights and offsets
  for(i in 2:length(n))
  {
    w[[i-1]] <- matrix((sample(100, n[i-1] * n[i], T) - 50) * 0.01 , n[i], n[i-1])
    b[[i-1]] <- matrix((sample(100, n[i], T) - 50) * 0.01, n[i], 1)
  }
  
  
#run gradient descent optimization  
  vals <- NN_optimize(w, b, XTrain, YTrain, alpha, num_iters, type)
  
#get predictions and accuracy for training examples
  pred_Train <- as.matrix(NN_predict(vals$w, vals$b, XTrain, type), nrow = 1)
  accuracy_Train <- 1 - sum(abs(t(YTrain) - pred_Train)) / length(YTrain)
    
  NNModel <- list("w" = vals$w, "b" = vals$b, "costs" = vals$costs, "activation" = type, "Train_Per" = accuracy_Train, "Train_Vals" = pred_Train)
  
#get predictions and accuracy for testing examples
  if(!is.null(XTest) && !is.null(YTest))
  {
    XTest <- t(as.matrix(XTest))
    YTest <- as.matrix(YTest)
    pred_Test <- as.matrix(NN_predict(vals$w, vals$b, XTest, type), nrow = 1)
    accuracy_Test <- 1 - sum(abs(t(YTest) - pred_Test)) / length(YTest)
    NNModel[["Test_Per"]] = accuracy_Test
    NNModel[["Test_Vals"]] = pred_Test 
  }
  return(NNModel)
}

'%+%' <- dget('matrixBroadcasting/plus.R')

#Run existing model against a new dataset
Predict_SNN <- function(NNModel, XTest, YTest)
{
  pred <- NN_predict(NNModel$w, NNModel$b, XTest, YTest, NNModel$activation)
  accuracy_Test <- 1 - sum(abs(t(YTest) - pred_Test)) / length(YTest)
  predModel <- list("Values" = pred, "Accuracy" = accuracy_Test)
}

#Get prediction results for a set of parameters and data
NN_predict <- function(w, b, XTest, type)
{
  z1 <- (w[[1]] %*% XTest)  %+% b[[1]]
  
  a1 <- activation(z1 , type)
  
  z2 <- (w[[2]] %*% a1) %+% b[[2]]
  
  a2 <- activation(z2, type)
  
  return(a2)
}

#Non-linear activation functions for determining classifications based on input
activation <- dget('activation.R')

#Generate a sample model trained to recognize the type of flower in the iris sample set.
# "setosa" = 1, "versicolor" = 2, "virginica" = 3
NN_Sample <- function(train_size = 100, n_h = 5, alpha = 0.01, num_iters = 10, act= "ReLU", type = "", raw = T)
{
  if(train_size > 140)
  {
    train_size <- 140
  }
  else if(train_size < 40)
  {
    train_size <- 40
  }
     
  train <- sort(sample(150, train_size))
  test <- 1:150
  test <- test[!(test %in% train)]
  xTrain <- as.matrix(iris[train, 1:4])
  yTrain <- as.matrix(as.numeric(iris[train, 5]))
  xTest <- as.matrix(iris[test, 1:4])
  yTest <- as.matrix(as.numeric(iris[test, 5]))
  NNMod <- NeuralNetwork_Model(XTrain = xTrain, YTrain = yTrain, XTest = xTest, YTest = yTest, n_h = n_h, alpha = alpha, num_iters = num_iters, type = act)
  
  if(raw)
  {
    vals <- NNMod$Model['Train_Vals']
    vals <- ifelse(vals < 2 & abs(2-vals) < abs(1-vals), 1, ifelse(vals < 2, 2, ifelse(abs(2-vals) < abs(3-vals), 2, 3)))
    NNMod$Model['Train_Vals'] <- vals
    vals <- NNMod$Model['Test_Vals']
    vals <- ifelse(vals < 2 & abs(2-vals) < abs(1-vals), 1, ifelse(vals < 2, 2, ifelse(abs(2-vals) < abs(3-vals), 2, 3)))
    NNMod$Model['Test_Vals'] <- vals
  }
  
  if(type == "")
  {
    return(list("XTrain" = xTrain, "YTrain" = yTrain, "XTest" = xTest, "YTest" = yTest, "Model" = NNMod))
  }
  else
  {
    return(NNMod[[type]])
  }
}
