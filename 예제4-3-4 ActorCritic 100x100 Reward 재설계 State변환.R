
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



######################
coord<-function(state){
  re_index<-which(state==1)
  xx<-ceiling(re_index/ 100) ## 행
  yy<-re_index %% 100  ## 열
  yy<-ifelse(yy ==0,100,yy)
  c(xx,yy)
}


stm<-matrix(1:10000,ncol=100,nrow=100,byrow=T)
action<-c("left","right","down","up")

state_size <-ncol(stm)*nrow(stm)


move<-function(x,action){
  
  if(action == "left"){
    if(x[2]-1<1){
      x
    }else{
      x[2]<-x[2]-1
      x
    }
  }
  if(action == "right"){
    if(x[2]+1>ncol(stm)){
      x
    }else{
      x[2]<-x[2]+1
      x
    }
  }
  if(action == "up"){
    if(x[1]-1<1){
      x
    }else{
      x[1]<-x[1]-1
      x
    }
  }
  if(action == "down"){
    if(x[1]+1>nrow(stm)){
      x
    }else{
      x[1]<-x[1]+1
      x
    }
  }
  x
}


next_where<-function(index){ 
  zero<-rep(0,10000)
  zero[index]<-1
  zero
}


wv<-rep(0,nrow(stm))
yv<-rep(0,ncol(stm))

convert_coord<-function(x){
  wv2<-wv
  yv2<-yv
  wv2[x[1]]<-1
  yv2[x[2]]<-1
  c(wv2,yv2)
}
state_size<-length(c(wv,yv))



##수정한 Reward 
return_reward<-function(next_state,current_state){
  re_index<-which(next_state==1)
  
  if(re_index==10000){
    reward<- 10# episode end
    done<-T
  }else if(sum(c(3001:3040,7031:7100) %in% re_index) ==1){
    reward<- -5
    done<-F
  }else{
    reward <- -0.1
    done<-F
  }
  if(re_index==which(current_state==1)){
    reward<-reward*2
  }
  reward<-reward -sqrt(sum((c(100,100)-st)^2))*0.01
  if(step==500){
    done<-T
  }
  
  c(reward,done)
  
}

### initialize neural network


# Actor Network  
{
  
  
  input_dim<-state_size
  hidden<-c(30)
  output_dim<-4
  size <- c(input_dim, hidden, output_dim)
  activationfun<-"relu"
  output<-"softmax"
  
  momentum<-0
  learningrate_scale<-1
  hidden_dropout = 0
  visible_dropout = 0
  learningrate<-0.0001
  
  
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
  qn1<- list(input_dim = input_dim, output_dim = output_dim, 
             hidden = hidden, size = size, activationfun = activationfun, 
             learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
             hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
             output = output, W = W, vW = vW, B = B, vB = vB)
  
  
  
  
  
}



# Critic Network  
{
  
  input_dim<-state_size
  hidden<-c(30)
  output_dim<-1
  size <- c(input_dim, hidden, output_dim)
  activationfun<-"relu"
  output<-"linear"
  
  batchsize<-30
  momentum<-0
  learningrate_scale<-0.99
  hidden_dropout = 0
  visible_dropout = 0
  numepochs = 10
  learningrate<-0.0001
  
  
  vW <- list()
  vB <- list()
  W <- list()
  B <- list()
  
  
  
  for (i in 2:length(size)) {
    W[[i - 1]] <- matrix(rnorm(size[i] * size[i - 1],0,2/input_dim),
                         c(size[i], size[i - 1]))
    B[[i - 1]] <- runif(size[i], min = -0.1, max = 0.1)
    vW[[i - 1]] <- matrix(rep(0, size[i] * size[i - 1]), 
                          c(size[i], size[i - 1]))
    vB[[i - 1]] <- rep(0, size[i])
  }
  vg<- list(input_dim = input_dim, output_dim = output_dim, 
            hidden = hidden, size = size, activationfun = activationfun, 
            learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
            hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
            output = output, W = W, vW = vW, B = B, vB = vB)
  
  
  
  
}




epoch<-50
init_data<-convert_coord(c(1,1))
dis_f<-0.99
reward_list<-c()
final_action_list<-list()
step_list<-c()
q_table<-list()
r<-1



for(i in 5001:10000){
  
  
  
  total_r<-0 ## total reward
  episode_done<-0 
  
  qn1<-nn.ff(qn1,t(init_data))
  step<-1
  action_list<-NULL
  st<-c(1,1)
  memory<-list()
  epoch_index<-1
  
  while(episode_done==0){
    
    if(step >1){
      qn1<-nn.ff(qn1,t(convert_next_state))
      action_prob<-qn1$post[[length(size)]]
      current_state<-next_state
    }else{
      current_state<- next_where(c(1,1))
      action_prob<-qn1$post[[length(size)]]
      
    }
    
    
    next_action<-  action[sample(1:4,1,prob=action_prob)]
    
    action_list<-c(action_list,next_action)
    convert_cur_state<-convert_coord(st)
    st<-move(st,next_action)
    state_index<-stm[st[1],st[2]]
    
    
    next_state<-next_where(state_index)
    re_ep<-return_reward(next_state,current_state) 
    convert_next_state<-convert_coord(st)
    
    total_r<-total_r+re_ep[1]
    episode_done<-re_ep[2]
    
    
    memory[[epoch_index]]<-list(input=(convert_cur_state),next_state=convert_next_state,action=next_action,done=re_ep[2],reward=re_ep[1])
    epoch_index<-epoch_index+1
    
    if(episode_done | step %% epoch == 0){

      rz<-1
      buffer_v<-c()
      if(episode_done){
        vs<-0
      }else{
        vs<-  nn.ff(vg,t(convert_next_state))$post[[length(size)]]
      }

      
      for(rz in (length(memory)):1){
        vs<-memory[[rz ]]$reward +dis_f*vs
        buffer_v<-c(buffer_v,vs)
      }
      
      
      buffer_v<-buffer_v[length(buffer_v):1]
      
      x_stack<-t(sapply(memory,function(x){x$input}))
      action_stack<-t(sapply(memory,function(x){x$action}))
      vg<-nn.ff(vg,(x_stack))
      
      td_error<-buffer_v-vg$post[[length(size)]]
      vg$e<-td_error
      vg<-nn.bp(vg)
      
      
      qn1<- nn.ff(qn1,(x_stack))
      yy<-qn1$post[[length(size)]]
      for(ml in 1:nrow(yy)){
        yy[ml,action %in% action_stack[ml]]<-(yy[ml,action %in%action_stack[ml]]+0.0001-1)
        
      }
      exp_v<- -yy * as.numeric(td_error)
      qn1$e<-  exp_v
      qn1<-nn.bp(qn1)
      memory<-list()
      epoch_index<-1
    }
    
    
    
    if(step == 500 |episode_done==1){
      cat("\n",i," epsode-",step) 
      step_list<-c(step_list,step)
      final_action_list[[i]]<-action_list
      reward_list<-c(reward_list,total_r)
      
      cat("\n final location")
      print(coord(next_state))
      ts.plot(reward_list,main=paste0("A2C Reward 재설계 + State ECV"))
      break;
    }
    
    step<-step+1
    
  }
  
  
  
  
  
  
}







setwd("D:\\개인폴더\\책\\R로하는 강화학습\\코드")
write.csv(reward_list,"A2C Reward 재설계 state ecv.csv",row.names = F)

base<-read.csv("A2C Reward 재설계.csv")



pl1<-c()
pl2<-c()
  
for(j in 1:(length(reward_list)-10)){
  
  pl1[j]<-mean(reward_list[j:(j+10)])
  pl2[j]<-mean(base[j:(j+10),1])  
  
  
}


ts.plot(cbind(pl1,pl2),col=c("red","blue"))

