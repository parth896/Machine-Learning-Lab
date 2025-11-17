import numpy as np
def loss(y,y_pred,n):
  return (1/(2*n))*np.sum((y-y_pred)**2)

def linear_regression(x_train,y_train,x_val,y_val,epochs=10000,lr=0.01,delta=0.000001):
  x_train=x_train.T
  x_val=x_val.T
  x_train = np.vstack((np.ones((1, x_train.shape[1])), x_train))
  x_val= np.vstack((np.ones((1, x_val.shape[1])), x_val))
  wt=np.zeros(x_train.shape[0])
  n=x_train.shape[1]
  while epochs>0:
    y_train_pred=x_train.T@wt
    y_val_pred=x_val.T@wt
    train_del=(-1/n)*(x_train@(y_train-y_train_pred))
    wt=wt-lr*(train_del)
    train_loss=loss(y_train,y_train_pred,n)
    val_loss=loss(y_val,y_val_pred,n)
    if abs(train_loss-val_loss)<=delta:
      break
    epochs-=1
  return wt
