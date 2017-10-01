function(z, type = c("sigmoid", "tanH", "ReLU"), deriv = F, n = 1)
{
  if(!deriv)
  {
    if(type[n] == "sigmoid"){return(1 / (1 + exp(-z)))}
    
    if(type[n] == "tanH"){return(tanh(z))}
    
    if(type[n] == "ReLU"){return(ifelse(z > 0, z, 0.01*z))}
    return(ifelse(z >= 0, 1, 0))
  }else
  {
    if(type[n] == "sigmoid"){return(activation(z, type[n]) * (1 - activation(z, type[n])))}
    
    if(type[n] == "tanH"){return(1 - tanh(z)^2)}
    
    if(type[n] == "ReLU"){return(ifelse(z > 0, z, 0.01*z)/ifelse(z == 0, 1e-6, z))}
    return(0)
  }
}