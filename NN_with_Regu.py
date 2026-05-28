import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons,weights_regulizer_l1=0,biases_regulizer_l1=0,weights_regulizer_l2=0,biases_regulizer_l2=0):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        self.weights_regulizer_l1=weights_regulizer_l1
        self.weights_regulizer_l2=weights_regulizer_l2
        self.biases_regulizer_l1=biases_regulizer_l1
        self.biases_regulizer_l2=biases_regulizer_l2
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(self.inputs,self.weights)+self.biases
    def backward(self,d_values):
        self.dweights=np.dot(self.inputs.T,d_values)
        self.dbiases=np.sum(d_values,axis=0,keepdims=True)
        
        
        if self.weights_regulizer_l1>0:
            dL1=np.ones_like(self.weights)
            dL1[self.weights<0]= -1
            self.dweights+=self.weights_regulizer_l1*dL1

        if self.weights_regulizer_l2>0:
            self.dweights+= (2*self.weights_regulizer_l2*self.weights)

        if self.biases_regulizer_l1>0:
            dL1=np.ones_like(self.biasess)
            dL1[self.biases<0]=-1
            self.dbiases+=self.biases_regulizer_l1*dL1

        if self.biases_regulizer_l2>0:
            self.dbiases+= (2*self.biases_regulizer_l2*self.biases)

        self.dinputs=np.dot(d_values,self.weights.T)


class Activation_ReLu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
        
    def backward(self,d_values):
        self.dinputs=d_values.copy()
        self.dinputs[self.inputs<=0]=0

class Activation_Softmax:
    def forward(self,inputs):
        self.inputs=inputs
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probablities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probablities
        
    def backward(self,d_values):
        self.dinputs=np.empty_like(d_values)
        for index,(single_output,single_dvalue) in enumerate(zip(self.output,d_values)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix,single_dvalue)



class Optimizer_SGD:
    def __init__(self,learning_rate=1.,decay=0.,momentum=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.momentum=momentum
        self.iterrations=0
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate * (1./(1.+self.decay*self.iterrations))
    def update_para(self,layer):
        if self.momentum:
            if not hasattr(layer,"weights_momentum"):
                layer.weights_momentum=np.zeros_like(layer.weights)
                layer.biases_momentum=np.zeros_like(layer.biases)
            weights_update=self.momentum*layer.weights_momentum - self.current_learning_rate*layer.dweights
            layer.weights_momentum=weights_update
            biases_update=self.momentum*layer.biases_momentum - self.current_learning_rate*layer.dbiases
            layer.biases_momentum=biases_update

        else:
            weights_update=-self.current_learning_rate*layer.dweights
            biases_update=-self.current_learning_rate*layer.dbiases
        layer.weights+=weights_update
        layer.biases+=biases_update
    def post_update_para(self):
        self.iterrations+=1

class Optimizer_Adagrad:
    def __init__(self,learning_rate=1.,decay=0.,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.iterrations=0
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate * (1./(1.+self.decay*self.iterrations))
    def update_para(self,layer):
        
            if not hasattr(layer,"weights_momentum"):
                layer.weights_cache=np.zeros_like(layer.weights)
                layer.biases_cache=np.zeros_like(layer.biases)
            layer.weights_cache+=layer.dweights**2
            layer.biases_cache+=layer.dbiases**2
            layer.weights+= -self.current_learning_rate * \
                            layer.dweights / \
                            (np.sqrt(layer.weights_cache)+self.epsilon)
            layer.biases+= -self.current_learning_rate * \
                            layer.dbiases / \
                            (np.sqrt(layer.biases_cache)+self.epsilon)
        
          
    def post_update_para(self):
        self.iterrations+=1
        
class Optimizer_RMSprop:
    def __init__(self,learning_rate=1.,decay=0.,epsilon=1e-7,rho=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.rho=rho
        self.iterrations=0
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate * (1./(1.+self.decay*self.iterrations))
    def update_para(self,layer):
        
            if not hasattr(layer,"weights_momentum"):
                layer.weights_cache=np.zeros_like(layer.weights)
                layer.biases_cache=np.zeros_like(layer.biases)
            layer.weights_cache=self.rho*layer.weights_cache+(1-self.rho)*layer.dweights**2
            layer.biases_cache=self.rho*layer.biases_cache+(1-self.rho)*layer.dbiases**2
            layer.weights+= -self.current_learning_rate * \
                            layer.dweights / \
                            (np.sqrt(layer.weights_cache)+self.epsilon)
            layer.biases+= -self.current_learning_rate * \
                            layer.dbiases / \
                            (np.sqrt(layer.biases_cache)+self.epsilon)
            
        
          
    def post_update_para(self):
        self.iterrations+=1

class Optimizer_Adam:
    def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-7,momentum=0.,decay=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.iterations=0
        self.momentum=momentum
        self.epsilon=epsilon
    def pre_update_para(self):
        
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+(self.decay*self.iterations)))
    def update_para(self,layer):
        if not hasattr(layer,"weights_cache"):
            layer.weights_momentum=np.zeros_like(layer.weights)
            layer.biases_momentum=np.zeros_like(layer.biases)
            layer.weights_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
        layer.weights_momentum=self.beta_1*layer.weights_momentum+(1-self.beta_1)*layer.dweights
        
        layer.biases_momentum=self.beta_1*layer.biases_momentum+(1-self.beta_1)*layer.dbiases
       

        weights_momentum_corrected=layer.weights_momentum/(1-(self.beta_1**(self.iterations+1)))
        biases_momentum_corrected=layer.biases_momentum/(1-(self.beta_1**(self.iterations+1)))

        layer.weights_cache=self.beta_2*layer.weights_cache+(1-self.beta_2)*layer.dweights**2
        
        layer.biases_cache=self.beta_2*layer.biases_cache+(1-self.beta_2)*layer.dbiases**2
        

        weights_cache_corrected=layer.weights_cache/(1-(self.beta_2**(self.iterations+1)))
        biases_cache_corrected=layer.biases_cache/(1-(self.beta_2**(self.iterations+1)))

        layer.weights+= -self.current_learning_rate * \
                        (weights_momentum_corrected) / \
                        (np.sqrt(weights_cache_corrected)+self.epsilon)
        layer.biases+= -self.current_learning_rate * \
                        (biases_momentum_corrected) / \
                        (np.sqrt(biases_cache_corrected)+self.epsilon)
        
    def post_update_para(self):
        self.iterations+=1

class Loss:
    def regularization_loss(self,layer):
        regularization_loss=0
        if layer.weights_regulizer_l1>0:
            regularization_loss+= layer.weights_regulizer_l1*(np.sum(np.abs(layer.weights)))
        if layer.weights_regulizer_l2>0:
            regularization_loss+= layer.weights_regulizer_l2*(np.sum(layer.weights*layer.weights))

        if layer.biases_regulizer_l1>0:
            regularization_loss+=layer.biases_regulizer_l1*(np.sum(np.abs(layer.biases)))

        if layer.biases_regulizer_l2>0:
            regularization_loss+=layer.biases_regulizer_l2*(np.sum(layer.biases*layer.biases))

        return regularization_loss    
    
    def calculate(self,y_pred,y_true):
        self.loss_per_sample=self.forward(y_pred,y_true)
        data_loss=np.mean(self.loss_per_sample)
        return data_loss

class LossCatogoricalEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidenses=y_clipped[range(samples),y_true]
        else:
            correct_confidenses=np.sum(y_clipped*y_true,axis=1)

        neg_likelyhoods=-np.log(correct_confidenses)
        return neg_likelyhoods

class Activation_Softmax_LossCatagoricalEntropy:
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=LossCatogoricalEntropy()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,d_values,y_true):
        sample=len(d_values)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=d_values.copy()
        self.dinputs[range(sample),y_true]+= -1
        self.dinputs=self.dinputs/sample

X,y=spiral_data(samples=100,classes=3)


dense1=Layer_Dense(2,64,weights_regulizer_l2=5e-4,biases_regulizer_l2=5e-4)
dense2=Layer_Dense(64,3)

activation1=Activation_ReLu()

loss_activation=Activation_Softmax_LossCatagoricalEntropy()

optimizer=Optimizer_Adam(learning_rate=0.02,decay=5e-7)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss=loss_activation.forward(dense2.output,y)


    regularization_loss=loss_activation.loss.regularization_loss(dense1)+loss_activation.loss.regularization_loss(dense2)

    loss=data_loss+regularization_loss
    y_true=y
    prediction=np.argmax(loss_activation.output,axis=1)
    if len(y_true.shape)==2:
        y_true=np.argmax(y_true,axis=1)
    acc=np.mean(prediction==y_true)    

    if not epoch%100:
        print(f'epoch : {epoch}',
            f' acc : {acc:.3f}',
            f' loss :{loss:.3f}',
            f' data_loss :{data_loss:.3f}',
            f' regu_loss :{regularization_loss:.3f}'
            f' lr :{optimizer.current_learning_rate}')
        

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_para()
    optimizer.update_para(dense1)
    optimizer.update_para(dense2)
    optimizer.post_update_para()

#for validation test
X_test,y_test=spiral_data(samples=100,classes=3)



dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss=loss_activation.forward(dense2.output,y_test)

y_true=y_test
prediction=np.argmax(loss_activation.output,axis=1)
if len(y_true.shape)==2:
    y_true=np.argmax(y_true,axis=1)
acc=np.mean(prediction==y_true) 


print(f'loss : {loss:.3f}',
    f'acc : {acc:.3f}')






        