
## relu 함수
relu<-function(x){
  ifelse(x>0,x,0)
}

swish<-function(x){
  x*sigm(x)
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
    else if (nn$activationfun == "swish") {
      
      nn$post[[i]] <- swish(nn$pre[[i]])
      
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
    else if (nn$activationfun == "swish") {
      d_act <-    (exp(-nn$post[[i]])*(nn$post[[i]]+1)+1)/((1+exp(-nn$post[[i]]))^2)
      
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
  xx<-ceiling(re_index/ 20) ## 행
  yy<-re_index %% 20  ## 열
  yy<-ifelse(yy ==0,20,yy)
  c(xx,yy)
}


stm<-matrix(1:400,ncol=20,nrow=20,byrow=T)
action<-c("left","right","down","up")


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
  zero<-rep(0,400)
  zero[index]<-1
  zero
}


wv<-rep(0,nrow(stm))
yv<-rep(0,ncol(stm))



convert_coord<-function(x,bonus_index,penalty_index,condition,stage_n){
  wv2<-wv
  yv2<-yv
  wv2[x[1]]<-1
  yv2[x[2]]<-1
  
  z<-(x-bonus_index)/1
  z2<-(x-penalty_index)/1
  
  
  bonusx<-wv
  bonusy<-yv
  
  bonusx[abs(z[1])]<-1
  bonusy[abs(z[2])]<-1
  
  if(z[1]>0){
    absx<-1
  }else{
    absx<-0
  }
  if(z[2]>0){
    absy<-1
  }else{
    absy<-0
  }
  
  penaltyx<-wv
  penaltyy<-yv
  
  penaltyx[abs(z2[1])]<-1
  penaltyy[abs(z2[2])]<-1
  if(z2[1]>0){
    abspx<-1
  }else{
    abspx<-0
  }
  if(z2[2]>0){
    abspy<-1
  }else{
    abspy<-0
  }
  stv<-rep(0,5)
  stv[stage_n]<-1
  
  c(wv2,yv2,absx,absy,bonusx,bonusy,abspx,abspy,penaltyx,penaltyy,condition,stv)
  # c(wv2,yv2,z,z2,condition)
  
}





return_reward<-function(goal_point,st,condition,stage_n){
  
  if(sum((st-goal_point)^2)< 2 & condition ==T){
    reward<-100
    done<-F
    if(stage_n == 4){
      done<-T
    }
    stage_n<-stage_n+1
  }else{
    
    reward<- -0.5
    done<-F
    if(stage_n==2){
      
    }
  }
  
  if(step==300){
    done<-T
    
  }
  
  if(sum((st-bonus_index)^2)< 2 & condition == F){
    reward<-reward+20
    condition<-T
  }
  if(sum((st-penalty_index)^2)< 2){
    reward<-reward-10
  }
  
  
  
  c(reward,done,condition,stage_n)
  
}




goal_point<-c(20,20)
bonus_index<-sample(2:20,2,replace = T)
penalty_index<-sample(2:20,2,replace = T)
init_data<-convert_coord(c(1,1),bonus_index,penalty_index,F,1)

state_size<-length(init_data)
### initialize neural network

# Actor Network  
{
  
  
  input_dim<-state_size
  hidden<-c(100)
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
  hidden<-c(100)
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



# Predictor Network

{
  
  input_dim<-(state_size)
  hidden<-c(100)
  output_dim<-10                                                   
  size2 <- c(input_dim, hidden, output_dim)
  activationfun<-"relu"
  output<-"linear"  
  
  batchsize<-30
  momentum<-0
  learningrate_scale<-0.9999
  hidden_dropout = 0
  visible_dropout = 0
  numepochs = 10
  learningrate<-0.005
  
  vW <- list()
  vB <- list()
  W <- list()
  B <- list()  
  
  for (i in 2:length(size2)) {
    W[[i - 1]] <- matrix(rnorm(size2[i] * size2[i - 1],0,2/input_dim),c(size2[i], size2[i - 1]))
    B[[i - 1]] <- runif(size2[i], min = -0.1, max = 0.1)
    vW[[i - 1]] <- matrix(rep(0, size2[i] * size2[i - 1]),c(size2[i], size2[i - 1]))
    vB[[i - 1]] <- rep(0, size2[i])
    
  }
  
  pn<- list(input_dim = input_dim, output_dim = output_dim, 
            hidden = hidden, size = size2, activationfun = activationfun, 
            learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
            hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
            output = output, W = W, vW = vW, B = B, vB = vB)
  
  
}

# Target Network
{  
  
  input_dim<-(state_size)
  hidden<-c(100)
  output_dim<-10                                                   
  size2 <- c(input_dim, hidden, output_dim)
  activationfun<-"relu"
  output<-"linear"
  
  batchsize<-30
  momentum<-0
  learningrate_scale<-0.9999
  hidden_dropout = 0
  visible_dropout = 0
  numepochs = 10
  learningrate<-0.005
  
  vW <- list()
  vB <- list()
  W <- list()
  B <- list()
  
  for (i in 2:length(size2)) {
    W[[i - 1]] <- matrix(rnorm(size2[i] * size2[i - 1],0,2/input_dim),c(size2[i], size2[i - 1]))
    B[[i - 1]] <- runif(size2[i], min = -0.1, max = 0.1)
    vW[[i - 1]] <- matrix(rep(0, size2[i] * size2[i - 1]),c(size2[i], size2[i - 1]))
    vB[[i - 1]] <- rep(0, size2[i])
    
  }
  tn<- list(input_dim = input_dim, output_dim = output_dim, 
            hidden = hidden, size = size2, activationfun = activationfun, 
            learningrate = learningrate, momentum = momentum, learningrate_scale = learningrate_scale, 
            hidden_dropout = hidden_dropout, visible_dropout = visible_dropout, 
            output = output, W = W, vW = vW, B = B, vB = vB)
  
}

epoch<-50
dis_f<-0.99
reward_list<-c()
final_action_list<-list()
step_list<-c()
replay_buffer<-list()
inr_list<-c() 
total_reward_list<-c()
bi<-1
st_list<-list()
par(mfrow=c(1,3))

stage_list<-c()
stage_step_list<-list()
for(i in 1:10000){
  
  bonus_index<-c(15,12)
  penalty_index<-c(5,4)
  stage_n<-1
  condition<-F
  goal_point<-c(20,20)
  init_data<-convert_coord(c(1,1),bonus_index,penalty_index,condition,1)
  episode_done<-0 
  qn1<-nn.ff(qn1,t(init_data))
  step<-1
  action_list<-NULL
  episode_buffer<-list()
  st<-c(1,1)
  memory<-list()
  epoch_index<-1
  total_r<-0 
  total_inr<-0
  total_reward<-0
  bi<-1
  epoch<-10
  
  st_2<-NULL
  bonus_index
  
  stage_step<-1
  
  while(episode_done==0){
    
    
    
    if(step >1){
      qn1<-nn.ff(qn1,t(convert_next_state))
      action_prob<-qn1$post[[length(size)]]
      
    }else{
      current_state<- next_where(c(1,1))
      action_prob<-qn1$post[[length(size)]]      
    }    
    
    next_action<-  action[sample(1:4,1,prob=action_prob)]   
    action_list<-c(action_list,next_action)
    convert_cur_state<-convert_coord(st,bonus_index,penalty_index,condition,stage_n)
    st_2<-rbind(st_2,st)
    st<-move(st,next_action)
    
    
    bf_stage<-stage_n
    
    re_ep<-return_reward(goal_point,st,condition,stage_n) 
    stage_n<-re_ep[4]
    condition<-re_ep[3]
    
    
    episode_done<-re_ep[2]
    
    
    if(episode_done){
      convert_next_state<-convert_coord(st,bonus_index,penalty_index,condition,bf_stage)    
    }else{
      if(bf_stage < stage_n){
        stage_step<-c(stage_step,step)
        if(stage_n==2){
          bonus_index<-c(4,17)
          penalty_index<-c(10,10)
          condition<-F
          st<-c(1,1)
        }
        if(stage_n==3){
          bonus_index<-c(3,8)
          penalty_index<-c(15,7)
          condition<-F
          st<-c(10,1)
        }
        if(stage_n==4){
          bonus_index<-c(10,8)
          penalty_index<-c(15,3)
          condition<-F
          st<-c(1,20)
          goal_point<-c(20,1)
        }
        convert_next_state<-convert_coord(st,bonus_index,penalty_index,condition,stage_n)
      }else{
        convert_next_state<-convert_coord(st,bonus_index,penalty_index,condition,stage_n)
      }
    }
    pn<-nn.ff(pn,t(convert_next_state)) 
    pf<-pn$post[[length(size2)]] 
    tn<-nn.ff(tn,t(convert_next_state))                                          
    tf<-tn$post[[length(size2)]]                       
    inr<- sum(abs(tf-pf))*100
    # inr<-0
    
    pn$e<-as.matrix(tf)-as.matrix(pf)
    pn<-nn.bp(pn)    
    
    total_inr<-total_inr+inr
    total_r<-total_r+re_ep[1]
    total_reward<-total_reward+inr+re_ep[1]
    
    
    memory[[epoch_index]]<- list(input=(convert_cur_state),next_state=convert_next_state,action=next_action,done=re_ep[2],reward=re_ep[1]+inr)
    episode_buffer[[bi]]<-  list(input=convert_cur_state,action=next_action,reward=re_ep[1]+inr,done=re_ep[2],next_state=convert_next_state,step=step)
    bi<-bi+1
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
      
      qn1$e<- ( exp_v)
      
      qn1<-nn.bp(qn1)
      memory<-list()
      epoch_index<-1
    }
    
    if(step == 300 |episode_done==1){
      cat("\n",i," epsode-",step) 
      step_list<-c(step_list,step)
      final_action_list[[i]]<-action_list
      reward_list[i]<-total_r
      inr_list[i]<-total_inr
      total_reward_list[i]<-total_reward
      st_list[[i]]<-st_2
      stage_list[i]<-stage_n
      stage_step_list[[i]]<-stage_step
      cat("\n final location")
      print(st)
      cat("\n stage",stage_n)
      
      if(i %% 10==0){
        ts.plot(reward_list,main="Extrinsic Reward")
        ts.plot(inr_list[-1],main="Intrinsic Reward")      
        
        ts.plot(total_reward_list,main="Total Reward")
        ts.plot(stage_list,main="Stage")
      }  
      break;
    }
    step<-step+1    
  }
  condition
  
  
  buffer_v<-c()
  if(episode_done){
    vs<-0
  }else{
    vs<-  nn.ff(vg,t(convert_next_state))$post[[length(size)]]
  } 
  
  for(rz in (length(episode_buffer)):1){    
    vs<-episode_buffer[[rz ]]$reward +dis_f*vs
    buffer_v<-c(buffer_v,vs)    
  }
  buffer_v<-buffer_v[length(buffer_v):1]
  
  action_stack<-t(sapply(episode_buffer,function(x){x$action}))
  x_stack<-lapply(episode_buffer,function(x){x$input})
  x_stack2<-do.call("rbind",x_stack)
  
  vg<-nn.ff(vg,(x_stack2))
  td_error<-buffer_v-vg$post[[length(size)]]  
  
  replay_buffer$input<-append(replay_buffer$input,x_stack)
  replay_buffer$action<-c(replay_buffer$action,action_stack)
  replay_buffer$reward<-c(replay_buffer$reward,buffer_v)
  replay_buffer$td_error<-c(replay_buffer$td_error,td_error)  
  
  if(i %% 20 ==0){
    
    batch_sample<-30
    batch_num<-100    
    prio_prob<-abs(replay_buffer$td_error)
    prio_prob<-prio_prob/sum(prio_prob)
    
    for(t in 1:batch_num){      
      sam<-sample(1:length(replay_buffer$reward),batch_sample,prob = prio_prob)
      x_stack<-do.call("rbind",replay_buffer$input[sam])
      reward_stack<-replay_buffer$reward[sam]
      
      vg<-nn.ff(vg,(x_stack))
      sil_td<-reward_stack-vg$post[[length(size)]]
      sil_index<-  sil_td> 0
      sil_td[!sil_index]<-0
      vg$e<-as.matrix(sil_td)
      vg<-nn.bp(vg)      
      
      qn1<- nn.ff(qn1,(x_stack))
      action_stack<-replay_buffer$action[sam]
      yy<-qn1$post[[length(size)]]
      
      for(ml in 1:nrow(yy)){
        yy[ml,action %in% action_stack[ml]]<-(yy[ml,action %in%action_stack[ml]]+0.0001-1)
        
      }
      
      
      exp_v<- -yy * as.numeric(sil_td)
      qn1$e<- exp_v
      qn1<-nn.bp(qn1)      
    }   
    
    if(length(replay_buffer$input)  >100000){
      ere<-length(replay_buffer$input)-100000      
      replay_buffer$input<-replay_buffer$input[-c(1: ere)]
      replay_buffer$action<-replay_buffer$action[-c(1: ere)]
      replay_buffer$reward<- replay_buffer$reward[-c(1: ere)]
      replay_buffer$td_error<- replay_buffer$td_error[-c(1: ere)]
      
    }
    
    
  }
}

avg_reward<-c()
avg_step<-c()
avg_stage<-c()

for(k in 1:(length(reward_list)-100)){
  avg_reward[k]<-mean(reward_list[k:(k+100)])
  avg_step[k]<- mean(step_list[k:(k+100)])
  avg_stage[k]<- mean(stage_list[k:(k+100)])
  
}


ts.plot(avg_stage,main="Stage 이동 평균 Plot")
ts.plot(avg_reward,main="Reward 이동 평균 Plot")
ts.plot(avg_step,main="Step 이동 평균 Plot")








plot_f(st_list[[which.max(reward_list)]])
st_list[[which.max(reward_list)]]
obs<-st_list[[which.max(reward_list)]]

# install.packages("plotly")
library(plotly)

obs<-data.frame(obs)
axx <- list(
  nticks = 4,
  range = c(0,20)
)

axy <- list(
  nticks = 8,
  range = c(0,20)
)

axz <- list(
  nticks = 4,
  range = c(0,10)
)
#

makec<-function(x){
  
  c(2*cos(x),2*sin(x),0)
  
}

coorcir<-t(sapply(1:360,makec))

cfs1<-matrix(rep(c(penalty_index,0),360),ncol=3,byrow=T)+coorcir
cfs2<-matrix(rep(c(bonus_index,0),360),ncol=3,byrow=T)+coorcir




index<-10000

obs2<-st_list[[index]]
sts<-stage_step_list[[index]]

stage_n<-2
if(stage_n==1){
  obs<-obs2[1:(sts[2]),]
  bonus_index<-c(15,12)
  penalty_index<-c(5,4)
  goal_point<-c(20,20)
  st<-c(1,1)
  
}

if(stage_n==2){
  obs<-obs2[(sts[2]+1):(sts[3]),]
  bonus_index<-c(4,17)
  penalty_index<-c(10,10)
  
  goal_point<-c(20,20)
  st<-c(1,1)
  
}
if(stage_n==3){
  obs<-obs2[(sts[3]+1):(sts[4]),]
  bonus_index<-c(3,8)
  penalty_index<-c(15,7)
  
  goal_point<-c(20,20)
}
if(stage_n==4){
  obs<-obs2[(sts[4]+1):nrow(obs2),]
  bonus_index<-c(10,8)
  penalty_index<-c(15,3)
  
  goal_point<-c(20,1)
}

cc<-1:nrow(obs)
obs<-data.frame(obs)


cfs1<-matrix(rep(c(penalty_index,0),360),ncol=3,byrow=T)+coorcir
cfs2<-matrix(rep(c(bonus_index,0),360),ncol=3,byrow=T)+coorcir


fig <- plot_ly(obs,x=obs[1,1], y= obs[1,2],z=1,type = 'scatter3d', mode = 'lines',name="Path",
               line = list(width = 4, color = ~cc, colorscale = list(c(0,'#BA52ED'), c(1,'#FCB040')))) %>%
  layout(autosize =F, aspectmode = 'manual', scene = list(xaxis=axx,yaxis=axy,zaxis=axz, aspectratio = list(x = 2, y = 2, z = 0.2))) %>% 
  add_trace(x=penalty_index[1], y= penalty_index[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Hole",
            line = list(width = 6, color = '#004e66'),
            marker = list(size = 10, color = "#004e66", cmin = -20, cmax = 50)) %>%
  add_trace(x=bonus_index[1], y= bonus_index[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Bonus",
            line = list(width = 6, color = '#fcbe32'),
            marker = list(size = 10, color = "#fcbe32", cmin = -20, cmax = 50 )) %>%
  add_trace(x=obs[1,1], y= obs[1,2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Start",
            line = list(width = 6, color = '#ff5f2e'),
            marker = list(size = 10, color = "#ff5f2e", cmin = -20, cmax = 50)) %>%
  add_trace(x=goal_point[1], y= goal_point[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Goal",
            line = list(width = 6, color = '#5e5e5f'),
            marker = list(size = 10, color = "#5e5e5f", cmin = -20, cmax = 50)) %>% 
  
  add_trace(x=cfs1[,1], y= cfs1[,2], z=cfs1[,3],mode="lines",line = list(width=1,color = '#004e66')) %>% 
  
  add_trace(x=cfs2[,1], y= cfs2[,2], z=cfs1[,3],mode="lines",line = list(width=1,color = '#fcbe32')) %>% 
  
  add_trace(x=goal_point[1], y= goal_point[2], z=1,type = "scatter3d", mode="text", text = "Goal", inherit=FALSE) %>% 
  add_trace(x=obs[1,1], y= obs[1,2], z=1,type = "scatter3d", mode="text", text = "Start", inherit=FALSE) %>% 
  add_trace(x=bonus_index[1], y= bonus_index[2], z=1,type = "scatter3d", mode="text", text = "Bonus", inherit=FALSE) %>% 
  add_trace(x=penalty_index[1], y= penalty_index[2], z=1,type = "scatter3d", mode="text", text = "Hole", inherit=FALSE)

fig



plot_f<-function(obs2,stage_n,sts){
  
  
  if(stage_n==1){
    obs<-obs2[1:(sts[2]),]
    bonus_index<-c(15,12)
    penalty_index<-c(5,4)
    goal_point<-c(20,20)
    st<-c(1,1)
    
  }
  
  if(stage_n==2){
    obs<-obs2[(sts[2]+1):(sts[3]),]
    bonus_index<-c(4,17)
    penalty_index<-c(10,10)
    
    goal_point<-c(20,20)
    st<-c(1,1)
    
  }
  if(stage_n==3){
    obs<-obs2[(sts[3]+1):(sts[4]),]
    bonus_index<-c(3,8)
    penalty_index<-c(15,7)
    
    goal_point<-c(20,20)
  }
  if(stage_n==4){
    obs<-obs2[(sts[4]+1):nrow(obs2),]
    bonus_index<-c(10,8)
    penalty_index<-c(15,3)
    
    goal_point<-c(20,1)
  }
  
  cc<-1:nrow(obs)
  obs<-data.frame(obs)
  
  
  cfs1<-matrix(rep(c(penalty_index,0),360),ncol=3,byrow=T)+coorcir
  cfs2<-matrix(rep(c(bonus_index,0),360),ncol=3,byrow=T)+coorcir
  
  
  fig <- plot_ly(obs,x=obs[,1], y= obs[,2],z=1,type = 'scatter3d', mode = 'lines',name="Path",
                 line = list(width = 4, color = ~cc, colorscale = list(c(0,'#BA52ED'), c(1,'#FCB040')))) %>%
    layout(autosize =F, aspectmode = 'manual', scene = list(xaxis=axx,yaxis=axy,zaxis=axz, aspectratio = list(x = 2, y = 2, z = 0.2))) %>% 
    add_trace(x=penalty_index[1], y= penalty_index[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Hole",
              line = list(width = 6, color = '#004e66'),
              marker = list(size = 10, color = "#004e66", cmin = -20, cmax = 50)) %>%
    add_trace(x=bonus_index[1], y= bonus_index[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Bonus",
              line = list(width = 6, color = '#fcbe32'),
              marker = list(size = 10, color = "#fcbe32", cmin = -20, cmax = 50 )) %>%
    add_trace(x=obs[1,1], y= obs[1,2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Start",
              line = list(width = 6, color = '#ff5f2e'),
              marker = list(size = 10, color = "#ff5f2e", cmin = -20, cmax = 50)) %>%
    add_trace(x=goal_point[1], y= goal_point[2], z=1,type = 'scatter3d',mode = 'lines+markers',name="Goal",
              line = list(width = 6, color = '#5e5e5f'),
              marker = list(size = 10, color = "#5e5e5f", cmin = -20, cmax = 50)) %>% 
    
    add_trace(x=cfs1[,1], y= cfs1[,2], z=cfs1[,3],mode="lines",line = list(width=1,color = '#004e66')) %>% 
    
    add_trace(x=cfs2[,1], y= cfs2[,2], z=cfs1[,3],mode="lines",line = list(width=1,color = '#fcbe32')) %>% 
    
    add_trace(x=goal_point[1], y= goal_point[2], z=1,type = "scatter3d", mode="text", text = "Goal", inherit=FALSE) %>% 
    add_trace(x=obs[1,1], y= obs[1,2], z=1,type = "scatter3d", mode="text", text = "Start", inherit=FALSE) %>% 
    add_trace(x=bonus_index[1], y= bonus_index[2], z=1,type = "scatter3d", mode="text", text = "Bonus", inherit=FALSE) %>% 
    add_trace(x=penalty_index[1], y= penalty_index[2], z=1,type = "scatter3d", mode="text", text = "Hole", inherit=FALSE)
  
  fig
}

plot_f(obs2,1,sts)
