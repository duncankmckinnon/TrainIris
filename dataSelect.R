source('kc_house_data.csv')
function(dataset = "Iris", set = NULL)
{
  moddata <- switch(EXPR = dataset,
                    "Iris" = list('data' = iris, 'xcol' = c(1:4), 'ycol' = c(5)),
                    "King County Housing" = list(kc_house),
                    "car" = cars,
                    "freeny" = freeny)
}