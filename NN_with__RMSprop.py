import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from entire_NN import Actiavtion_ReLU,Activation_Softmax_Loss_CatagoricalEntropyLoss,Layer_Dense
class Optimizer_RSMprop:
    def __init__(self,momentum=0.,learning_rate=0.001,rho=0.9,epsilon=1e-7,decay=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.momentum=momentum
        self.epsilon=epsilon
        self.rho=rho
        self.iterations=0
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate * \
                                        1/(1+(self.decay*self.iterations))

    def update_para(self,layer):
        if not hasattr(layer,"weights_cache"):
            layer.weights_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
        
        layer.weights_cache= self.rho*layer.weights_cache + (1-self.rho)*layer.dweights**2
        layer.biases_cache=self.rho*layer.biases_cache + (1-self.rho)*layer.dbiases**2

        layer.weights+= -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weights_cache)+self.epsilon)
        
        layer.biases+= -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.biases_cache)+self.epsilon)

        
    def post_update_para(self):
        self.iterations+=1

X,y=spiral_data(samples=100,classes=3)
dense1=Layer_Dense(2,64)
activation1=Actiavtion_ReLU()
dense2=Layer_Dense(64,3)
loss_activation=Activation_Softmax_Loss_CatagoricalEntropyLoss()

optimizer=Optimizer_RSMprop(learning_rate=0.001,decay=1e-7,rho=0.9)
for epoch in range(20001):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    loss=loss_activation.forward(dense2.outputs,y)

    y_true=y
    predictions=np.argmax(loss_activation.output,axis=1)
    if len(y_true.shape)==2:
        y_true=np.argmax(y_true,axis=1)
    acc=np.mean(predictions==y_true)

    if not epoch%100:
        print(f"epoch :{epoch}" + f" acc :{acc:.3f}" + f" loss :{loss:.3f}" + f" lr :{optimizer.current_learning_rate}")

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_para()
    optimizer.update_para(dense1)
    optimizer.update_para(dense2)
    optimizer.post_update_para()





        