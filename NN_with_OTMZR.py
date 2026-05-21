import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()


class Loss:
    def calculate(self,y_pred,y_true):
        self.loss_per_sample=self.forward(y_pred,y_true)
        loss_data=np.mean(self.loss_per_sample)
        return loss_data
class Loss_Catagorical_Entropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        self.y_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_evidenses=self.y_clipped[range(samples),y_true]
        else:
            correct_evidenses=np.sum(y_pred*y_true,axis=1)
        return -np.log(correct_evidenses)    
class Activation_softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probablities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probablities    
class Layer_dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(self.inputs,self.weights)+self.biases
    def backward(self,d_values):
        self.dweights=np.dot(self.inputs.T,d_values)
        self.dbiases=np.sum(d_values,axis=0,keepdims=True)
        self.dinputs=np.dot(d_values,self.weights.T)    

class Activation_Relu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,self.inputs)
    def backward(self,d_values):
        self.dinputs=d_values.copy()
        self.dinputs[self.inputs<=0]=0    

class Activation_softmax_LossCatagoricalEntropy():
    def __init__(self):
        self.activation=Activation_softmax()
        self.loss=Loss_Catagorical_Entropy()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,y_pred,y_true):
        samples=len(y_pred)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=y_pred.copy()
        self.dinputs[range(samples),y_true]-=1    
        self.dinputs=self.dinputs/samples
class Optimizer_SGD:
    def __init__(self,learning_rate=1):
        self.learning_rate=learning_rate

    def update_para(self,layer):
        layer.weights += -self.learning_rate*layer.dweights
        layer.biases += -self.learning_rate*layer.dbiases


X,y=spiral_data(samples=100,classes=3)

dense1=Layer_dense(2,64)

activation1=Activation_Relu()

dense2=Layer_dense(64,3)

loss_activation=Activation_softmax_LossCatagoricalEntropy()


optimizer=Optimizer_SGD()

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss=loss_activation.forward(dense2.output,y)

    predictions=np.argmax(loss_activation.output,axis=1)
    if len(y.shape)==2:
        y_true=np.argmax(y,axis=1)
    acc=np.mean(predictions==y)

    if not epoch%100:
        print(f"epoch : {epoch} ,accuracy : {acc:.3f},loss : {loss:.3f}")

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_para(dense1)
    optimizer.update_para(dense2)    