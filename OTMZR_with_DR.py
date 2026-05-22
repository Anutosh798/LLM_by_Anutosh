from entire_NN import Layer_Dense,Actiavtion_ReLU,Activation_Softmax_Loss_CatagoricalEntropyLoss
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

class Optimizer_SGD:
    def __init__(self,learning_rate=1.,decay=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iteration=0
    def pre_update_para(self):
         if self.decay:
            self.current_learning_rate=self.learning_rate *\
                                        (1./(1.+(self.decay*self.iteration)))
    def update_para(self,layer):
        layer.weights += -self.current_learning_rate*layer.dweights
        layer.biases += -self.current_learning_rate*layer.dbiases
    def post_update_para(self):
        self.iteration+=1

X,y=spiral_data(samples=100,classes=3)

dense1=Layer_Dense(2,64)
activation1=Actiavtion_ReLU()
dense2=Layer_Dense(64,3)
loss_activation=Activation_Softmax_Loss_CatagoricalEntropyLoss()

optimizer=Optimizer_SGD(decay=1e-3)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    loss=loss_activation.forward(dense2.outputs,y)

    predictions=np.argmax(loss_activation.output,axis=1)
    if len(y.shape)==2:
        y=np.argmax(y,axis=1)
    acc=np.mean(predictions==y)

    optimizer.pre_update_para()
    if not epoch%100:
        print(f"epoch : {epoch}"+f" loss :{loss:.3f}"+ f" accuracy :{acc:.3f}" + f"lr:{optimizer.current_learning_rate}")


    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_para()
    optimizer.update_para(dense1)
    optimizer.update_para(dense2)
    optimizer.post_update_para()


