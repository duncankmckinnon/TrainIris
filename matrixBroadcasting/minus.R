
#NumPy style matrix subtraction with broadcasting '%-%'
function(m1, m2)
{
  '%+%' <- dget('matrixBroadcasting/plus.R')
  return(m1 %+% -m2)
}