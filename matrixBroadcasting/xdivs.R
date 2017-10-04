
#NumPy style matrix division with broadcasting '%d%'
function(m1, m2)
{
  '%x%' <- dget('matrixBroadcasting/xtimes.R')
  return(m1 %x% (1/m2))
}