#this is file where i try out snippets
import numpy as np

# a=np.array([[1,2],
#             [9,4]])

# b=np.argmax(a,axis=1)
# print(b)
#trying out how is that possible where "if a:" means "if a != 0",
# a=0
# if a:
#     print("hello")
# else:
#     print("bye")    

#trying out how np.clip() works

a=np.array([[1,2,5],
            [-1,8,3]])
a_clipped=np.clip(a,1,5)

print(a_clipped)