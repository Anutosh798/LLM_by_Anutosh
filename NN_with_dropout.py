import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from NN_with_Regu import Layer_Dense,Activation_ReLu,Activation_Softmax_LossCatagoricalEntropy,Loss,Optimizer_Adam
nnfs.init()



class Layer_dropout:
    def __init__(self,rate):
        self.rate=1-rate
    def forward(self,inputs):
        self.inputs=inputs
        self.mask=np.random.binomial(1,self.rate,size=inputs.shape)/self.rate
        self.output=self.inputs*self.mask
    def backward(self,d_values):
        self.dinputs=d_values*self.mask
        
X,y=spiral_data(samples=1000,classes=3)


dense1=Layer_Dense(2,512,weights_regulizer_l2=5e-4,biases_regulizer_l2=5e-4)
dense2=Layer_Dense(512,3)

activation1=Activation_ReLu()
dropout1=Layer_dropout(0.1)
loss_activation=Activation_Softmax_LossCatagoricalEntropy()

optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
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
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
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

