
goRead <- function()
{
  kc_house_data <- read.csv('kc_house_data.csv')[,c(3,4:8,11:14,20:21)]
  iris_data <- iris
  return(list("King County Housing" = kc_house_data, "Iris" = iris))
  
}
