from entire_NN import Actiavtion_ReLU,Layer_Dense,Activation_Softmax_Loss_CatagoricalEntropyLoss
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


class Optimizer_SGD:
    def __init__(self,learning_rate=1.,decay=0.,momentum=0):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.momentum=momentum
        self.iteration=0
        
    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*\
                                        1/(1.+(self.decay*self.iteration))
    def update_para(self,layer):
        if self.momentum:
            if not hasattr(layer,"weights_momentums"):
                layer.weights_momentums=np.zeros_like(layer.weights)
                layer.biases_momentums=np.zeros_like(layer.biases)

            weights_update=(self.momentum*layer.weights_momentums)-(self.current_learning_rate*layer.dweights)
            layer.weights_momentums=weights_update

            biases_update=(self.momentum*layer.biases_momentums)-(self.current_learning_rate*layer.dbiases)
            layer.biases_momentums=biases_update

        else:
            weights_update= -self.current_learning_rate*layer.dweights
            biases_update= -self.current_learning_rate*layer.dbiases        

        layer.weights+=weights_update
        layer.biases+=biases_update
    def post_update_para(self):
        self.iteration+=1


X,y=spiral_data(samples=300,classes=3)

dense1=Layer_Dense(2,64)
activation1=Actiavtion_ReLU()
dense2=Layer_Dense(64,3)
loss_activation=Activation_Softmax_Loss_CatagoricalEntropyLoss()

optimizer=Optimizer_SGD(decay=1e-3,momentum=0.9)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    loss=loss_activation.forward(dense2.outputs,y)

    predictions=np.argmax(loss_activation.output,axis=1)
    y_true=y
    if len(y_true.shape)==2:
        y_true=np.argmax(y_true,axis=1)
    acc=np.mean(predictions==y_true)    
    optimizer.pre_update_para()
    if not epoch%100:
        print(f"epoch :{epoch}"+f" loss :{loss:.3f}"+f" acc :{acc:.3f}"+f" lr :{optimizer.current_learning_rate}")  





    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # optimizer.pre_update_para()
    optimizer.update_para(dense1)
    optimizer.update_para(dense2)
    optimizer.post_update_para()