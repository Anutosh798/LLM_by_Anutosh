import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()
X,y=spiral_data(samples=100,classes=3)
# plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
# plt.show()
class Loss:
    def calculate(self,y_pred,y_true):
        self.loss_per_sample=self.forward(y_pred,y_true)
        mean_loss=np.mean(self.loss_per_sample)
        return mean_loss
class LossCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        self.y_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences=self.y_clipped[range(samples),y_true]
        else:
            correct_confidences=np.sum(self.y_clipped*y_true,axis=1)
        return -np.log(correct_confidences)
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
class Activtation_ReLu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,self.inputs)
    def backward(self,d_values):
        self.dinputs=d_values.copy()
        self.dinputs[self.inputs<=0]=0
class Activation_Softmax_LossCrossEntropy:
    def __init__(self):
        self.activation=Activation_softmax()
        self.loss=LossCrossEntropy()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,d_values,y_true):
        samples=len(d_values)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=d_values.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples


class Optimizer_AdaGrad:
    def __init__(self,momentum=0.,learning_rate=1.,decay=0.,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.momentum=momentum
        self.epsilon=epsilon
        self.iteration=0
        
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate *\
                                        1/(1+(self.decay*self.iteration))
    def update_para(self,layer):
        if not hasattr(layer,"weights_cache"):
            layer.weights_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)

        layer.weights_cache+= layer.dweights**2
        layer.biases_cache+=layer.dbiases**2

        layer.weights+= -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weights_cache) + self.epsilon)
        layer.biases+= -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.biases_cache) + self.epsilon)    
    def post_update_para(self):
        self.iteration+=1
        
X,y=spiral_data(samples=300,classes=3)

dense1=Layer_dense(2,64)
activation1=Activtation_ReLu()
dense2=Layer_dense(64,3)
loss_activation=Activation_Softmax_LossCrossEntropy()

optimizer=Optimizer_AdaGrad(decay=1e-4)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss=loss_activation.forward(dense2.output,y)

    predictions=np.argmax(loss_activation.output,axis=1)
    y_true=y
    if len(y_true.shape)==2:
        y_true=np.argmax(y_true,axis=1)
    acc=np.mean(predictions==y_true)    
    # optimizer.pre_update_para()
    if not epoch%100:
        print(f"epoch :{epoch}"+f" loss :{loss:.3f}"+f" acc :{acc:.3f}"+f" lr :{optimizer.current_learning_rate}")  





    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_para()
    optimizer.update_para(dense1)
    optimizer.update_para(dense2)
    optimizer.post_update_para()


               