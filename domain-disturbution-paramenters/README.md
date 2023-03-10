Variational encoder parameters learning

# purpose
- through the diff. size of IO of NN, at the input, middle & output part, we wanna find out the para. of distribution generating the current samples

# philo
- the network structure size is decreasing to only keep the most important information, and hopefully we hope we can the most important information is the distribution parameter

# tests
- we will firstly use simple kernels like [1d gaussian], using its generated samples to predict its kernel parameters
- next we will use [z = exp(x)*sin(y)]
- with noise
- then we will use more complicated functions after that to test the theory

# usage
- compress the generated sample from a distribution to its kernel parameter

