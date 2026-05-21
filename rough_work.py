#this is file where i try out snippets
import numpy as np

a=np.array([[1,2],
            [9,4]])

b=np.argmax(a,axis=1)
print(b)