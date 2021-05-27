
## relu 함수
relu<-function(x){
  ifelse(x>0,x,0)
}


##network feedforward

nn.ff<-function (nn, batch_x) 
  
{
  m <- nrow(batch_x)
  
  if (nn$visible_dropout > 0) {
    nn$dropout_mask[[1]] <- dropout.mask(ncol(batch_x), nn$visible_dropout)
    batch_x <- t(t(batch_x) * nn$dropout_mask[[1]])
  }
  
  nn$post[[1]] <- batch_x
  
  for (i in 2:(length(nn$size) - 1)) {
    nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) + 
                       nn$B[[i - 1]])
    
    if (nn$activationfun == "sigm") {
      nn$post[[i]] <- sigm(nn$pre[[i]])
    }
    
    else if (nn$activationfun == "tanh") {
      nn$post[[i]] <- tanh(nn$pre[[i]])
    }
    else if (nn$activationfun == "relu") {
      nn$post[[i]] <- relu(nn$pre[[i]])
    }
    else if (nn$activationfun == "linear") {
      nn$post[[i]] <- (nn$pre[[i]])
    }
    else {
      stop("unsupport activation function!")
    }
    
    if (nn$hidden_dropout > 0) {
      nn$dropout_mask[[i]] <- dropout.mask(ncol(nn$post[[i]]), 
                                           nn$hidden_dropout)
      nn$post[[i]] <- t(t(nn$post[[i]]) * nn$dropout_mask[[i]])
    }
  }
  
  
  i <- length(nn$size)
  nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) + 
                     nn$B[[i - 1]])
  
  
  if (nn$output == "sigm") {
    nn$post[[i]] <- sigm(nn$pre[[i]])
    
  } else if (nn$output == "linear") {
    nn$post[[i]] <-  nn$pre[[i]] 
    
  } else if (nn$output == "softmax") {
    
    nn$post[[i]] <- exp(nn$pre[[i]])
    nn$post[[i]] <- nn$post[[i]]/rowSums(nn$post[[i]])
    
  }
  nn
  
}




## network back propagation
nn.bp<-function (nn) 
{
  n <- length(nn$size)
  d <- list()
  if (nn$output == "sigm") {
    d[[n]] <- -nn$e * (nn$post[[n]] * (1 - nn$post[[n]]))
    
  }
  else if (nn$output == "linear" || nn$output == "softmax") {
    d[[n]] <- -nn$e
  }
  
  for (i in (n - 1):2) {
    if (nn$activationfun == "sigm") {
      d_act <- nn$post[[i]] * (1 - nn$post[[i]])
    }
    else if (nn$activationfun == "tanh") {
      d_act <- 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn$post[[i]]^2)
    }
    else if (nn$activationfun == "relu") {
      d_act <-  ifelse(nn$post[[i]]>=0,1,0)
    }
    d[[i]] <- (d[[i + 1]] %*% nn$W[[i]]) * d_act
    if (nn$hidden_dropout > 0) {
      d[[i]] <- t(t(d[[i]]) * nn$dropout_mask[[i]])
    }
  }
  

  
  for (i in 1:(n - 1)) {
    dw <- t(d[[i + 1]]) %*% nn$post[[i]]/nrow(d[[i + 1]])
    dw <- dw * nn$learningrate
    
    if (nn$momentum > 0) {
      nn$vW[[i]] <- nn$momentum * nn$vW[[i]] + dw
      dw <- nn$vW[[i]]
    }
    
    nn$W[[i]] <- nn$W[[i]] - dw
    db <- colMeans(d[[i + 1]])
    db <- db * nn$learningrate
    
    if (nn$momentum > 0) {
      nn$vB[[i]] <- nn$momentum * nn$vB[[i]] + db
      db <- nn$vB[[i]]
    }
    
    nn$B[[i]] <- nn$B[[i]] - db
    
  }
  
  nn
  
}


sigm<-function (x) {
  1/(1 + exp(-x))
}



  
  input_dim<-4
  hidden<-c(30)
  output_dim<-3
  size <- c(input_dim, hidden, output_dim)
  activationfun<-"tanh"
  output<-"softmax"
  
  momentum<-0
  learningrate_scale<-1
  hidden_dropout = 0
  visible_dropout = 0
  learningrate<-0.01
  
  
  vW <- list()
  vB <- list()
  W <- list()
  B <- list()
  
  
  
  for (i in 2:length(size)) {
    W[[i - 1]] <- matrix(runif(size[i] * size[i - 1], 
                               min = -0.1, max = 0.1), c(size[i], size[i - 1]))
    B[[i - 1]] <- runif(size[i], min = -0.1, max = 0.1)
    vW[[i - 1]] <- matrix(rep(0, size[i] * size[i - 1]), 
                          c(size[i], size[i - 1]))
    vB[[i - 1]] <- rep(0, size[i])
  }
  nn<- list(input_dim = input_dim, output_dim = output_dim, 
             hidden = hidden, size = size, activationfun = activationfun, 
             learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
             hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
             output = output, W = W, vW = vW, B = B, vB = vB)
  
  
  
  
  
data(iris)

train_x<-iris[,-5]
# setosa
# versicolor
# virginica

y<-as.character(iris[,5])


mat<-matrix(0,nrow=nrow(train_x),nco=3)
mat[y=="setosa",1]<-1
mat[y=="versicolor",2]<-1
mat[y=="virginica",3]<-1


try<-apply(mat,1,which.max)

sam<-sample(1:nrow(train_x),nrow(train_x)*0.7)

for(i in 1:1000){
  
  nn<-nn.ff(nn,as.matrix(train_x[sam,]))
  nn$e<-mat[sam,] - nn$post[[3]] 
  nn<-nn.bp(nn)
  
  nn<-nn.ff(nn,as.matrix(train_x[-sam,]))
  acc<-sum(apply(nn$post[[3]],1,which.max) == try[-sam])/length(try[-sam])
  
  if(i %% 100 == 0){
  cat("\n",i,"epoch-validation accuracy : ",acc)  
  
  }
}
