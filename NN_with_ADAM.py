import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from entire_NN import Actiavtion_ReLU,Activation_Softmax_Loss_CatagoricalEntropyLoss,Layer_Dense
nnfs.init()
class Optimizer_ADAM:
    def __init__(self,learning_rate=1.,decay=0.,beta_1=0.9,beta_2=0.999,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.iterations=0
        self.beta_1=beta_1
        self.beta_2=beta_2

    def pre_update_para(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate * \
                                        1/(1+(self.decay*self.iterations))
            
    def update_para(self,layer):
        if not hasattr(layer,"biases_cache"):
            layer.weights_momentums=np.zeros_like(layer.weights)
            layer.biases_momentums=np.zeros_like(layer.biases)
            layer.weights_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
            

        #weights_momentums
        layer.weights_momentums=self.beta_1*layer.weights_momentums+(1-self.beta_1)*layer.dweights
        layer.biases_momentum=self.beta_1*layer.biases_momentums+(1-self.beta_1)*layer.dbiases

        #weights_momentums_corrected
        weights_momentums_corrected=layer.weights_momentums / (1 - self.beta_1**(self.iterations+1))
        biases_momentums_corrected=layer.biases_momentum / (1 - self.beta_1**(self.iterations+1))


        #weights_cache
        layer.weights_cache=self.beta_2*layer.weights_cache+(1-self.beta_2)*layer.dweights**2
        layer.biases_cache=self.beta_2*layer.biases_cache+(1-self.beta_2)*layer.dbiases**2



        #weights_cache_corrected
        weights_cache_corrected=layer.weights_cache / (1 - self.beta_2**(self.iterations+1))
        biases_cache_corrected=layer.biases_cache / (1 - self.beta_2**(self.iterations+1))

        #weights
        layer.weights+= -self.current_learning_rate * weights_momentums_corrected / (np.sqrt(weights_cache_corrected)+self.epsilon)
        layer.biases+= -self.current_learning_rate * biases_momentums_corrected / (np.sqrt(biases_cache_corrected)+self.epsilon)
    def post_update_para(self):
        self.iterations+=1

X,y=spiral_data(samples=100,classes=3)
dense1=Layer_Dense(2,64)
activation1=Actiavtion_ReLU()
dense2=Layer_Dense(64,3)
loss_activation=Activation_Softmax_Loss_CatagoricalEntropyLoss()

optimizer=Optimizer_ADAM(learning_rate=0.02,decay=1e-5)
for epoch in range(10001):
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
