def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    
    w1=np.random.randn(n_h,n_x)*0.01
    w2=np.random.randn(n_y,n_h)*0.01
    b1=np.zeros((n_h,1))
    b2=np.zeros((n_y,1))
    
    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    #np.random.seed(1)
    
    m=len(layer_dims)
    w={}
    b={}
    for i in range(1,m):
        w[i]=np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        b[i]=np.zeros((layer_dims[i],1))
    return w,b

def linear_forward(x,w,b):
    return np.dot(w,x)+b

def activation(z,act):
    if act=='sigmoid':
        return sigmoid(z)
    elif act=='relu':
        return relu(z)
    elif act=='tanh':
        return tanh(z) 
    

def l_layer_forward(x,w,b):
    
    l=len(w)
   
    z,a={},{}
    a[0]=x
    for i in range(1,l):
        z[i]=linear_forward(a[i-1],w[i],b[i])
        a[i]=activation(z[i],'relu')
    z[l]=linear_forward(a[l-1],w[l],b[l])
    a[l]=activation(z[l],'sigmoid')
    #a[l]=sigmoid(z[l])
    return z,a

def compute_cost(y_pre,y):
    m=y.shape[1]
    return -1/m*np.sum(y*np.log(y_pre)+(1-y)*np.log(1-y_pre))

def l_layer_backward(z,a,w,x,y):
    l=len(w)
    m=x.shape[1]
    delta,dw,db,da={},{},{},{}
    a[0]=x
    
    delta[l]=a[l]-y     #1*m
    for i in range(l-1,0,-1):
        da[i]=np.where(z[i]>0,1,0)
        delta[i]=np.dot(w[i+1].T,delta[i+1])*da[i]
        #delta[i]=np.dot(w[i+1].T,delta[i+1])*(1-np.square(a[i]))
    
    for i in range(1,l+1):
        db[i]=1/m*np.sum(delta[i],axis=1,keepdims=True)
        dw[i]=1/m*np.dot(delta[i],a[i-1].T)
    
    return dw,db

def update(w,b,dw,db,learn_rate):
    m=len(w)
    for i in range(1,m+1):
        w[i]=w[i]-learn_rate*dw[i]
        b[i]=b[i]-learn_rate*db[i]
    return w,b

def l_model(layer_dims,x,y,step,learn_rate):
    w,b=initialize_parameters_deep(layer_dims)
    l=len(layer_dims)-1
    
    c=[]
    for i in range(step):
        z,a=l_layer_forward(x,w,b)
        #print(a[2])
        cost=compute_cost(a[l],y)
        dw,db=l_layer_backward(z,a,w,x,y)
        w,b=update(w,b,dw,db,learn_rate)
        if(i%100==0):
            c.append(cost)
    return w,b,c

def predict(w,b,x,y):
    m=len(w)
    z,a=l_layer_forward(x,w,b)
    
    p=np.round(a[m])
    accuracy=np.sum(p==y)/y.shape[1]
    return accuracy

