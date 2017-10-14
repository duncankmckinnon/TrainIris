#Deep Neural Network Model (L-hidden layers of N_H[l] nodes)
#Duncan McKinnonx

source('matrixBroadcasting.R')
source('parseData.R')
source('activation.R')

Deep_NeuralNetwork_Model <- function(XTrain, YTrain, n_h = c(5,4,3), alpha = 0.01, num_iters = 10, type = "relu", XTest = NULL, YTest = NULL, regularize = F, momentum = F)
{
  #internal model function to perform gradient descent optimization on weights and offset
  Deep_NN_optimize <- function(w, b, XTrain, YTrain, nlayers, alpha, num_iters, type, momentum)
  {
    costs <- c()
    if(momentum)
    {
      b1 = 0.5
      b2 = 0.8
      Vdw <- list()
      Vdb <- list()
      Sdw <- list()
      Sdb <- list()
      for(i in 1:length(w))
      {
        Vdw[[i]] <- w[[i]] * 0
        Vdb[[i]] <- b[[i]] * 0
        Sdw[[i]] <- w[[i]] * 0
        Sdb[[i]] <- b[[i]] * 0
      }
      momentum_params <- list("momentum" = T, "Vdw" = Vdw, "Vdb" = Vdb, "Sdw" = Sdw, "Sdb" = Sdb, "b1" = b1, "b2" = b2)
    }
    else
    {
      momentum_params <- list("momentum" = F)
    }
    for(i in 1:num_iters)
    {
      vals <- Deep_NN_propagate(w, b, nlayers, XTrain, YTrain, type, momentum_params)
      for(j in 1:nlayers)
      {
        if(!momentum)
        {
          w[[j]] = w[[j]] - (alpha * vals$dw[[j]])
          b[[j]] = b[[j]] - (alpha * vals$db[[j]])
        }
        else
        {
          momentum_params$Vdw[[j]] <- vals$Vdw[[j]] / (1 - (momentum_params$b1 ^ i)) 
          momentum_params$Vdb[[j]] <- vals$Vdb[[j]] / (1 - (momentum_params$b1 ^ i)) 
          momentum_params$Sdw[[j]] <- vals$Sdw[[j]] / (1 - (momentum_params$b2 ^ i)) 
          momentum_params$Sdb[[j]] <- vals$Sdb[[j]] / (1 - (momentum_params$b2 ^ i)) 
          w[[j]] = w[[j]] - (alpha * momentum_params$Vdw[[j]] / sqrt(momentum_params$Sdw[[j]]))
          b[[j]] = b[[j]] - (alpha * momentum_params$Vdb[[j]] / sqrt(momentum_params$Sdb[[j]]))          
        }
      }
      costs <- c(costs, vals$cost)
    }
    return(list("w" = w, "b" = b, "dw" = vals$dw,  "db" = vals$db,  "costs" = costs))
  }
  
  #internal model function to perform forward propogation to get estimates based on current weights and offset
  # and back propogation for next optimization step  
  Deep_NN_propagate <- function(w, b, nlayers, XTrain, YTrain, type, momentum_params)
  {
    m <- dim(XTrain)[2]
    zn <- list()
    an <- list()
    da <- list()
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
      
    cost <- (1/m) * sum((YTrain - t(an[[nlayers]]))^2)
    dz[[nlayers]] <- an[[nlayers]] - t(YTrain)
    
    for(j in nlayers:2)
    {
      dw[[j]] <- (1/m) * dz[[j]] %*% t(an[[j-1]])
    
      db[[j]] <- (1/m) * colSums(t(dz[[j]]))
      
      if(momentum_params$momentum)
      {
        momentum_params$Vdw[[j]] <- momentum_params$b1 * momentum_params$Vdw[[j]] + (1 - momentum_params$b1) * dw[[j]]
        momentum_params$Vdb[[j]] <- momentum_params$b1 * momentum_params$Vdb[[j]] + (1 - momentum_params$b1) * db[[j]] 
        momentum_params$Sdw[[j]] <- momentum_params$b2 * momentum_params$Sdw[[j]] + (1 - momentum_params$b2) * dw[[j]]^2
        momentum_params$Sdb[[j]] <- momentum_params$b2 * momentum_params$Sdb[[j]] + (1 - momentum_params$b2) * db[[j]]^2
        
      }
      
      dz[[j-1]] <- (t(w[[j]]) %*% dz[[j]]) * activation(zn[[j-1]], type, deriv = T)
    }
    
    dw[[1]] <- (1/m) * dz[[1]] %*% t(XTrain)
    
    db[[1]] <- (1/m) * colSums(t(dz[[1]]))
    
    if(momentum_params$momentum)
    {
      momentum_params$Vdw[[1]] <- momentum_params$b1 * momentum_params$Vdw[[1]] + (1 - momentum_params$b1) * dw[[1]]
      momentum_params$Vdb[[1]] <- momentum_params$b1 * momentum_params$Vdb[[1]] + (1 - momentum_params$b1) * db[[1]] 
      momentum_params$Sdw[[1]] <- momentum_params$b2 * momentum_params$Sdw[[1]] + (1 - momentum_params$b2) * dw[[1]]^2
      momentum_params$Sdb[[1]] <- momentum_params$b2 * momentum_params$Sdb[[1]] + (1 - momentum_params$b2) * db[[1]]^2
      return(list("dw" = dw, "db" = db, "Vdw" = momentum_params$Vdw, "Vdb" = momentum_params$Vdb, "Sdw" = momentum_params$Sdw, "Sdb" = momentum_params$Sdb, "cost" = cost))
    }
    
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
    b[[i-1]] <- matrix(0, nvals[i], 1)
  }
  
  
  #run gradient descent optimization  
  vals <- Deep_NN_optimize(w, b, XTrain, YTrain, n, alpha, num_iters, type, momentum)
  
  #get predictions and accuracy for training examples
  pred_Train <- as.matrix(Deep_NN_predict(vals$w, vals$b, XTrain, n, type), nrow = 1)
  YTrain <- ifelse(is.na(YTrain), 0, YTrain)
  accuracy_Train <- 1 - sum((t(YTrain) - pred_Train) ^ 2) / length(YTrain)
  cor_Train <- cor.test(t(YTrain), pred_Train)$estimate
  
  NNModel <- list("w" = vals$w, "b" = vals$b, "costs" = vals$costs, "activation" = type, "Train_Per" = accuracy_Train, "Train_Cor" = cor_Train, "Train_Vals" = pred_Train)
  
  #get predictions and accuracy for testing examples
  if(!is.null(XTest) && !is.null(YTest))
  {
    XTest <- t(as.matrix(XTest))
    YTest <- as.matrix(YTest)
    pred_Test <- as.matrix(Deep_NN_predict(vals$w, vals$b, XTest, n, type), nrow = 1)
    YTest <- ifelse(is.na(YTest), 0, YTest)
    accuracy_Test <- 1 - sum((t(YTest) - pred_Test) ^ 2) / length(YTest)
    cor_Test <- cor.test(t(YTest), pred_Test)$estimate
    NNModel[["Test_Per"]] = accuracy_Test
    NNModel[["Test_Cor"]] = cor_Test
    NNModel[["Test_Vals"]] = pred_Test 
  }
  return(NNModel)
}

#Run existing model against a new dataset
Predict_DNN <- function(Deep_NNModel, XTest, YTest)
{
  pred <- Deep_NN_predict(Deep_NNModel$w, Deep_NNModel$b, XTest, Deep_NNModel$activation)
  YTest <- ifelse(is.na(YTest), 0, YTest)
  accuracy_Test <-  1 - sum((t(YTest) - pred_Test) ^ 2) / length(YTest)
  cor_Test <- cor.test(t(YTest), pred_Test)$estimate
  predModel <- list("Values" = pred, "Accuracy" = accuracy_Test, "Correlation" = cor_Test)
  return(predModel)
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

#Generate a sample model trained to recognize the type of flower in the iris sample set.
# "setosa" = 1, "versicolor" = 2, "virginica" = 3
Deep_NN_Sample <- function(data_set = iris, xcol = c(1:4), ycol = 5, train_size = 100, test_size = 50, n_h = c(5,4,3), alpha = 0.01, num_iters = 10,  act = "relu", type = "", raw = F, momentum = T)
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
  dataset$YTrain <- as.numeric(dataset$YTrain)
  dataset$YTest <- as.numeric(dataset$YTest)
  DNNMod <- Deep_NeuralNetwork_Model(XTrain = dataset$XTrain, YTrain = dataset$YTrain, XTest = dataset$XTest, YTest = dataset$YTest, alpha = alpha, num_iters = num_iters, n_h = n_h, type = act, momentum = momentum)
  
  if(!raw)
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
