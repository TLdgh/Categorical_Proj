df<-data.frame(x1=runif(320, 1, 5), x2=runif(320, -1,6), y=rbinom(320,1, 0.56))

head(df)

X_train <- scale(df[, c(1:2)])
y_train <- df$y

X_train <- as.matrix(X_train, byrow=TRUE)
X_train <- t(X_train)
y_train <- as.matrix(y_train, byrow=TRUE)
y_train <- t(y_train)


getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  n_x <- nrow(X)
  n_h <- hidden_neurons
  n_y <- nrow(y)
  
  size <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)
  
  return(size)
}

#initialize parameters
initializeParameters <- function(X, list_layer_size){
  
  m <- dim(X)[2]
  
  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  
  W1 <- matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, n_h * m), nrow = n_h, ncol = m)
  W2 <- matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, n_y * m), nrow = n_y, ncol = m)
  
  params <- list("W1" = W1,
                 "b1" = b1, 
                 "W2" = W2,
                 "b2" = b2)
  
  return (params)
}


sigmoid<-function(x){
  return(1/(1+exp(-x)))
}

#calculate output of each neuron
forwardPropagation <- function(X, params, list_layer_size){
  
  m <- dim(X)[2]
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
  Z1 <- W1 %*% X + b1
  A1 <- sigmoid(Z1)
  Z2 <- W2 %*% A1 + b2
  A2 <- sigmoid(Z2)
  
  fwd <- list("Z1" = Z1,
                "A1" = A1, 
                "Z2" = Z2,
                "A2" = A2)
  
  return (fwd)
}

# Calculate Binary Cross-Entropy loss
computeCost <- function(X, y, fwd_prop) {
  m <- dim(X)[2]
  A2 <- fwd_prop$A2
  logprobs <- (log(A2) * y) + (log(1-A2) * (1-y))
  cost <- -sum(logprobs/m)
  return (cost)
}

#Calculate partial derivatives
backwardPropagation <- function(X, y, fwd, params, list_layer_size){
  
  m <- dim(X)[2]
  
  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  
  A2 <- fwd$A2
  A1 <- fwd$A1
  W2 <- params$W2
  
  dZ2 <- A2 - y
  dW2 <- 1/m * (dZ2 %*% t(A1)) 
  db2 <- matrix(1/m * sum(dZ2), nrow = n_y)
  db2_new <- matrix(rep(db2, m), nrow = n_y)
  
  dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
  dW1 <- 1/m * (dZ1 %*% t(X))
  db1 <- matrix(1/m * sum(dZ1), nrow = n_h)
  db1_new <- matrix(rep(db1, m), nrow = n_h)
  
  grads <- list("dW1" = dW1, 
                "db1" = db1,
                "dW2" = dW2,
                "db2" = db2)
  
  return(grads)
}


# Gradient descent
updateParameters <- function(grads, params, learning_rate){
  
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
  dW1 <- grads$dW1
  db1 <- grads$db1
  dW2 <- grads$dW2
  db2 <- grads$db2
  
  
  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2
  
  updated_params <- list("W1" = W1,
                         "b1" = b1,
                         "W2" = W2,
                         "b2" = b2)
  
  return (updated_params)
}


MainModel <- function(X, y, num_iteration, hidden_neurons, lr){
  
  layer_size <- getLayerSize(X, y, hidden_neurons)
  init_params <- initializeParameters(X, layer_size)
  cost_history <- c()
  for (i in 1:num_iteration) {
    fwd_prop <- forwardPropagation(X, init_params, layer_size)
    cost <- computeCost(X, y, fwd_prop)
    back_prop <- backwardPropagation(X, y, fwd_prop, init_params, layer_size)
    update_params <- updateParameters(back_prop, init_params, learning_rate = lr)
    init_params <- update_params
    cost_history <- c(cost_history, cost)
    
    if (i %% 10000 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
  }
  
  model_out <- list("updated_params" = update_params,
                    "cost_hist" = cost_history)
  return (model_out)
}






