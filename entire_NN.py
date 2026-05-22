from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt

# X,y=spiral_data(samples=100,classes=3)

# plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
# plt.show()

class Loss:
    def calculate(self,predicted_outputs,target_outputs):
        self.loss_per_sample=self.forward(predicted_outputs,target_outputs)
        data_loss=np.mean(self.loss_per_sample)
        return data_loss
class Loss_Catagorical_Entropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        self.y_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidenses=self.y_clipped[range(samples),y_true]
        else:
            correct_confidenses=np.sum(self.y_clipped*y_true,axis=1)

        return -np.log(correct_confidenses)        
class Activation_softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probablities=(exp_values/np.sum(exp_values,axis=1,keepdims=True))
        self.output=probablities
        
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs=inputs
        self.outputs=np.dot(self.inputs,self.weights)+self.biases
    def backward(self,d_values):    
        self.dweights=np.dot(self.inputs.T,d_values)
        self.dbiases=np.sum(d_values,axis=0,keepdims=True)
        self.dinputs=np.dot(d_values,self.weights.T)
class Actiavtion_ReLU:
    def forward(self,inputs):
        self.inputs=inputs
        self.outputs=np.maximum(0,self.inputs)
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0


class Activation_Softmax_Loss_CatagoricalEntropyLoss:
    def __init__(self):
        self.activation=Activation_softmax()
        self.loss=Loss_Catagorical_Entropy()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output   
        return self.loss.calculate(self.output,y_true)
    def backward(self,d_values,y_true):
        samples=len(d_values)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=d_values.copy()
        self.dinputs[range(samples),y_true]=self.dinputs[range(samples),y_true]-1
        self.dinputs=self.dinputs/samples

X,y=spiral_data(samples=100,classes=3)
dense1=Layer_Dense(2,3)
relu_activation=Actiavtion_ReLU()
dense2=Layer_Dense(3,3)
loss_activation=Activation_Softmax_Loss_CatagoricalEntropyLoss()

dense1.forward(X)
relu_activation.forward(dense1.outputs)
dense2.forward(relu_activation.outputs)
loss=loss_activation.forward(dense2.outputs,y)


print(loss_activation.output[:5])
print("loss :",loss)

predictions=np.argmax(loss_activation.output,axis=1)
if len(y.shape)==2:
    y=np.argmax(y,axis=1)
accuracy=np.mean(predictions==y)
print("acc",accuracy)    


loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
relu_activation.backward(dense2.dinputs)
dense1.backward(relu_activation.dinputs)

print("dense1 weights",dense1.dweights)
print("dense1 biases",dense1.dbiases)
print("dense2 weights",dense2.dweights)
print("dense2 biases",dense2.dbiases)

