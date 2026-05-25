import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()


X,y=spiral_data(samples=100,classes=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
# plt.show()


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.n_inputs=n_inputs
        self.n_neurons=n_neurons
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(inputs,self.weights)+self.biases
    def backward(self,d_values):
        self.dweights=np.dot(self.inputs.T,d_values)
        self.dbiases=np.sum(d_values,axis=0,keepdims=True)
        self.dinputs=np.dot(d_values,self.weights.T)

class Activation_ReLu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,self.inputs)
    def backward(self,d_values):
        self.dinputs=d_values.copy()
        self.dinputs[self.inputs<=0]=0

class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probablities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probablities
        
class Loss:
    def calculate(self,y_pred,y_true):
        self.loss_per_sample=self.forward(y_pred,y_true)
        data_loss=np.mean(self.loss_per_sample)
        return data_loss

class Catagorical_Entropy_Loss(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        self.y_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidenses=self.y_clipped[range(samples),y_true]
        else:
            correct_confidenses=np.sum(self.y_clipped*y_true,axis=1)

        negative_likely_hoods=-np.log(correct_confidenses)
        return negative_likely_hoods        


class Activation_Softmax_Losscatogoricalentopy:
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Catagorical_Entropy_Loss()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
        
    def backward(self,y_pred,y_true):
        samples=len(y_pred)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=0)
        self.dinputs=y_pred.copy()
        self.dinputs[range(samples),y_true]-= 1
        self.dinputs=self.dinputs/samples

class Optimizer_ADAM:
    def __init__(self,learning_rate=0.001,decay=0.,beta_1=0.9,beta_2=0.999,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.decay=decay
        self.epsilon=epsilon
        self.iterations=0
        
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate= self.current_learning_rate * \
                                        1/(1+(self.decay,self.iterations))
    def update_para(self,layer):
        if not hasattr (layer,"weights_cache"):
            layer.weights_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
            layer.weights_momentums=np.zeros_like(layer.weights)
            layer.biases_momentums=np.zeros_like(layer.biases)

        layer.weights_momentums=self.beta_1*layer.weights_momentums+(1-self.beta_1)*layer.dweights
        layer.biases_momentums=self.beta_1*layer.biases_momentums+(1-self.beta_1)*layer.dbiases

        weights_momentums_corrected=layer.weights_momentums/ (1-self.beta_1**(self.iterations+1))
        biases_momentums_corrected=layer.biases_momentums/ (1-self.beta_1**(self.iterations+1)) 

        layer.weights_cache=self.beta_2*layer.weights_cache+(1-self.beta_2)*layer.dweights**2
        layer. biases_cache=self.beta_2*layer.biases_cache+(1-self.beta_2)*layer.dbiases**2

        weights_cache_corrected=layer.weights_cache/ (1-self.beta_2**(self.iterations+1))
        biases_cache_corrected=layer.biases_cache/ (1-self.beta_2**(self.iterations+1))

        layer.weights+= -self.current_learning_rate *(weights_momentums_corrected/(np.sqrt(weights_cache_corrected)+self.epsilon)) 
        layer.biases+= -self.current_learning_rate *(biases_momentums_corrected/(np.sqrt(biases_cache_corrected)+self.epsilon))

    def post_update_para(self):
        self.iterations+=1

optmizer=Optimizer_ADAM()
dense1=Layer_Dense(2,64)
activation1=Activation_ReLu()
dense2=Layer_Dense(64,3)
loss_activation=Activation_Softmax_Losscatogoricalentopy()

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss=loss_activation.forward(dense2.output,y)
    y_true=y
    predictions=np.argmax(loss_activation.output,axis=1)
    if len(y_true.shape)==2:
        y_true=np.argmax(y_true,axis=1)

    acc=np.mean(predictions==y_true)

    if not epoch%100:
        print(f"epoch :{epoch}" + f"acc :{acc:.3f}" + f"loss :{loss:.3f}" + f"lr :{optmizer.current_learning_rate}")


    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optmizer.post_update_para()
    optmizer.update_para(dense1)
    optmizer.update_para(dense2)    
    optmizer.pre_update_para()