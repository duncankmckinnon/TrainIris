parseModelData <- function(data_set, x_cols, y_cols, train_size, test_size = 0, scale_vals = F)
{
  tot_size <- dim(data_set)[1]
  
  if(test_size > 0)
  {
    sample_data <- sample(tot_size, (train_size + test_size), F)
    train <- sample_data[1:train_size]
    test <- sample_data[train_size + 1:length(sample_data)]
  }
  else
  {
    train <- sample(tot_size, train_size, F)
    test <- 1:tot_size
    test <- test[!(test %in% train)]
  }
  
  xTrain <- data_set[sort(train), x_cols]
  yTrain <- data_set[sort(train), y_cols]
  xTest <- data_set[sort(test), x_cols]
  yTest <- data_set[sort(test), y_cols]
  
  return(list("XTrain" = xTrain, "YTrain" = yTrain, "XTest" = xTest, "YTest" = yTest))
}