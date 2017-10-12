
#Non-linear activation functions for determining classifications based on input
activation <- function(z, type = c("sigmoid", "tanH", "ReLU"), deriv = F, n = 1, R_leaky = T)
{
  if(!deriv)
  {
    if(type[n] == "sigmoid"){return(g_sigmoid(z))}
    
    if(type[n] == "tanH"){return(g_tanh(z))}
    
    if(type[n] == "ReLU"){return(g_relu(z, leaky = R_leaky))}
    
    if(type[n] == "Leaky_ReLU"){return(g_relu(z))}
    return(ifelse(z >= 0, 1, 0))
  }else
  {
    if(type[n] == "sigmoid"){return(g_sigmoid(z, deriv = T))}
    
    if(type[n] == "tanH"){return(g_tanh(z, deriv = T))}
    
    if(type[n] == "ReLU"){return(g_relu(z, deriv = T, leaky = R_leaky))}
    
    if(type[n] == "Leaky_ReLU"){return(g_relu(z))}
    return(0)
  }
}

g_sigmoid <- function(z, deriv = F)
{
  if(deriv)
  {
    return(g_sigmoid(z) * (1 - g_sigmoid(z)))
  }
  return(1 / (1 + exp(-z)))
}

g_tanh <- function(z, deriv = F)
{
  if(deriv)
  {
    return(1 - tanh(z)^2)
  }
  return(tanh(z))
}

g_relu <- function(z, deriv = F, thresh = 0, leaky = T)
{
  if(deriv)
  {
    return(g_relu(z, thresh = thresh, leaky = leaky)/ifelse(z == 0, 1e-6, z))
  }
  return(ifelse(z >= thresh, z, ifelse(leaky, 0.01*z, 0)))
}