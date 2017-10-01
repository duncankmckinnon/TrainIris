#Deep Neural Network Model (L-hidden layers of N_H[l] nodes)
#Duncan McKinnonx


Deep_NeuralNetwork_Model <- function(XTrain, YTrain, n_h = c(5,4,3), alpha = 0.01, num_iters = 10, type = "tanH", XTest = NULL, YTest = NULL)
{
  #internal model function to perform gradient descent optimization on weights and offset
  Deep_NN_optimize <- function(w, b, XTrain, YTrain, nlayers, alpha, num_iters, type)
  {
    costs <- c()
    for(i in 1:num_iters)
    {
      vals <- Deep_NN_propagate(w, b, nlayers, XTrain, YTrain, type)
      for(j in 1:nlayers)
      {
        w[[j]] = w[[j]] - (alpha * vals$dw[[j]])
        b[[j]] = b[[j]] - (alpha * vals$db[[j]])
      }
      costs <- c(costs, vals$cost)
    }
    return(list("w" = w, "b" = b, "dw" = vals$dw,  "db" = vals$db,  "costs" = costs))
  }
  
  #internal model function to perform forward propogation to get estimates based on current weights and offset
  # and back propogation for next optimization step  
  Deep_NN_propagate <- function(w, b, nlayers, XTrain, YTrain, type)
  {
    m <- dim(XTrain)[2]
    zn <- list()
    an <- list()
    dz <- list()
    dw <- list()
    db <- list()
    
    zn[[1]] <- (w[[1]] %*% XTrain) %+% b[[1]]
    
    an[[1]] <- activation(zn[[1]], type)
    
    for(i in 2:nlayers)
    {
      j <- i-1
      zn[[i]] <- (w[[i]] %*% an[[j]]) %+% b[[i]]
    
      an[[i]] <- activation(zn[[i]], type)
    }
      
    cost <- (-1/m) * sum((YTrain - t(an[[nlayers]]))^2)
    dz[[nlayers]] <- an[[nlayers]] - t(YTrain)
    
    for(j in nlayers:2)
    {
      dw[[j]] <- (1/m) * dz[[j]] %*% t(an[[j-1]])
    
      db[[j]] <- (1/m) * colSums(t(dz[[j]]))
      
      dz[[j-1]] <- (t(w[[j]]) %*% dz[[j]]) * activation(zn[[j-1]], type, T)
    }
    
    dw[[1]] <- (1/m) * dz[[1]] %*% t(XTrain)
    
    db[[1]] <- (1/m) * colSums(t(dz[[1]]))
    
    
    return(list("dw" = dw, "db" = db, "cost" = cost))
  }
  
  #initialization of variables for training phase
  XTrain <- t(as.matrix(XTrain))
  YTrain <- as.matrix(YTrain)
  
  #number of inner layer dimensions  
  nvals <- c(dim(XTrain)[1], n_h, dim(YTrain)[2])
  n <- length(nvals) - 1
  
  w <- list()
  b <- list()
  
  #initialize 2 levels of weights and offsets
  for(i in 2:length(nvals))
  {
    w[[i-1]] <- matrix((sample(100, nvals[i-1] * nvals[i], T) - 50) * 0.01 , nvals[i], nvals[i-1])
    b[[i-1]] <- matrix((sample(100, nvals[i], T) - 50) * 0.01, nvals[i], 1)
  }
  
  
  #run gradient descent optimization  
  vals <- Deep_NN_optimize(w, b, XTrain, YTrain, n, alpha, num_iters, type)
  
  #get predictions and accuracy for training examples
  pred_Train <- as.matrix(Deep_NN_predict(vals$w, vals$b, XTrain, n, type), nrow = 1)
  accuracy_Train <- 1 - sum(abs(t(YTrain) - pred_Train)) / length(YTrain)
  
  NNModel <- list("w" = vals$w, "b" = vals$b, "costs" = vals$costs, "activation" = type, "Train_Per" = accuracy_Train, "Train_Vals" = pred_Train)
  
  #get predictions and accuracy for testing examples
  if(!is.null(XTest) && !is.null(YTest))
  {
    XTest <- t(as.matrix(XTest))
    YTest <- as.matrix(YTest)
    pred_Test <- as.matrix(Deep_NN_predict(vals$w, vals$b, XTest, n, type), nrow = 1)
    accuracy_Test <- 1 - sum(abs(t(YTest) - pred_Test)) / length(YTest)
    NNModel[["Test_Per"]] = accuracy_Test
    NNModel[["Test_Vals"]] = pred_Test 
  }
  return(NNModel)
}

#Run existing model against a new dataset
Predict_DNN <- function(Deep_NNModel, XTest, YTest)
{
  pred <- Deep_NN_predict(Deep_NNModel$w, Deep_NNModel$b, XTest, YTest, Deep_NNModel$activation)
  accuracy_Test <- 1 - sum(abs(t(YTest) - pred_Test)) / length(YTest)
  predModel <- list("Values" = pred, "Accuracy" = accuracy_Test)
}

#Get prediction results for a set of parameters and data
Deep_NN_predict <- function(w, b, XTest, layers, type)
{
  zn <- list()
  an <- list()

  zn[[1]] <- (w[[1]] %*% XTest)  %+% b[[1]]
  
  an[[1]] <- activation(zn[[1]] , type)
  
  for(i in 2:layers)
  {
    zn[[i]] <- (w[[i]] %*% an[[i-1]]) %+% b[[i]]
  
    an[[i]] <- activation(zn[[i]], type)
  }
  return(an[[layers]])
}

#matrix broadcasting addition
'%+%' <- dget('matrixBroadcasting/plus.R')

#Non-linear activation functions for determining classifications based on input
activation <- dget('activation.R')

#parse a dataset in a data frame into a training and sample set
parseModelData <- dget('parseData.R')

#Generate a sample model trained to recognize the type of flower in the iris sample set.
# "setosa" = 1, "versicolor" = 2, "virginica" = 3
Deep_NN_Sample <- function(train_size = 100, n_h = c(5,4,3), alpha = 0.01, num_iters = 10,  activation = "ReLU", type = "", raw = T)
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
  dataset$YTrain <- as.numeric(dataset$YTrain)
  dataset$YTest <- as.numeric(dataset$YTest)
  DNNMod <- Deep_NeuralNetwork_Model(XTrain = dataset$XTrain, YTrain = dataset$YTrain, XTest = dataset$XTest, YTest = dataset$YTest, alpha = alpha, num_iters = num_iters, n_h = n_h, type = activation)
  
  if(raw)
  {
    vals <- DNNMod[['Train_Vals']]
    vals <- ifelse(vals <= 1, 1, ifelse(vals < 2 & abs(2-vals) < abs(1-vals), 1, ifelse(vals <= 2, 2, ifelse(abs(2-vals) < abs(3-vals), 2, 3))))
    DNNMod[['Train_Vals']] <- vals
    vals <- DNNMod[['Test_Vals']]
    vals <- ifelse(vals <= 1, 1, ifelse(vals < 2 & abs(2-vals) < abs(1-vals), 1, ifelse(vals <= 2, 2, ifelse(abs(2-vals) < abs(3-vals), 2, 3))))
    DNNMod[['Test_Vals']] <- vals
  }
  
  if(type == "")
  {
    return(list("XTrain" = dataset$XTrain, "YTrain" = dataset$YTrain, "XTest" = dataset$XTest, "YTest" = dataset$YTest, "Model" = DNNMod))
  }
  else
  {
    return(DNNMod[[type]])
  }
}
