import numpy as np
import torch
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from fixedpoint import FixedPoint
import math

from LST_lib import LST_2d_one

input_size = 28
output_size = 10
model = LST_2d_one(input_size, output_size)

N_epoch = 300
model.load_state_dict(torch.load(f'model_backup/L2DST_1l_epoch_{N_epoch}.pth')) # , map_location=device

weights = model.state_dict()

def min_max_show(A, key):
    print(f'{key}: size = ', A[key].shape)
    print(f'{key}: max value = ', torch.max(A[key]))
    print(f'{key}: min value = ', torch.min(A[key]))

# Weights and biases analysis
min_max_show(weights, 'L2DST.dense1.weight')
min_max_show(weights, 'L2DST.dense1.bias')
min_max_show(weights, 'L2DST.dense2.weight')
min_max_show(weights, 'L2DST.dense2.bias')
min_max_show(weights, 'W_o.weight')
min_max_show(weights, 'W_o.bias')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

valset = datasets.MNIST('c:/Docs/GitHub/Datasets/', download=False, train=False, transform=transform)

img, label = valset[4]
img_ = img.view(img.size(0), -1).squeeze().numpy()

fig = plt.figure(figsize=(1,1))
plt.imshow(img_.reshape(28,-1))
# plt.colorbar()

e1, e2 = model.get_embeddings(img)

e1 = e1.squeeze(0)
e2 = e2.squeeze(0)

class MAC():
    def __init__(self, int_bits, frac_bits):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.acc = FixedPoint(0, signed=True, m=self.int_bits, n=self.frac_bits, rounding="down")
    
    def initialization(self, value):
        self.acc = FixedPoint(value, signed=True, m=self.int_bits, n=self.frac_bits, rounding="down")
        
    def run(self, a, b, q_bit_to_remove):
        p = a*b
        p = FixedPoint(float(p), signed=True, m=self.int_bits, n= (p.n - q_bit_to_remove), rounding="down")
        self.acc = self.acc + p
        self.acc = FixedPoint(float(self.acc), signed=True, m=self.int_bits, n = self.acc.n, rounding="down")
        
class MEM():
    def __init__(self, size, int_bits, frac_bits):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.size = size
        self.mem = [] # FixedPoint(np.zeros((size)), signed=True, m=self.int_bits, n=self.frac_bits)
        
    def initialization(self, array):
        for addr in range(len(array)):
            self.mem.append(FixedPoint(array[addr], signed=True, m=self.int_bits, n = self.frac_bits))
    
    def write_value(self, value, addr):
        self.mem[addr] = FixedPoint(float(value), signed=True, m=self.int_bits, n = self.frac_bits) 
        
    def numpy(self):
        np_arr = np.zeros((self.size))
        for addr in range(self.size):
            np_arr[addr] = float(self.mem[addr])
        
        return np_arr
        
class tanh():
    def __init__(self, int_bits, frac_bits):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.const_q_plus_one  = FixedPoint(1-2**(-frac_bits), signed=True, m=int_bits, n=frac_bits, str_base=16, rounding="down")
        self.const_q_minus_one = FixedPoint(-1, signed=True, m=int_bits, n=frac_bits, str_base=16, rounding="down")
        
    def calc(self, x):
        x_fp = FixedPoint(float(x), signed=True, m=self.int_bits, n = self.frac_bits, rounding="down")
        x_quater = x_fp >> 2
        if (x_fp>=2):
            mul_out = FixedPoint((1), signed=True, m=self.int_bits, n = self.frac_bits, str_base=16)
        elif (x<=-2):
            mul_out = FixedPoint((-1), signed=True, m=self.int_bits, n = self.frac_bits, str_base=16)
        else:
            if (x>0):
                add_sub_out = self.const_q_plus_one - x_quater
            else:
                add_sub_out = self.const_q_plus_one + x_quater
            mul_out = (add_sub_out*x_fp)
            mul_out.m = self.int_bits
            mul_out.n = self.frac_bits
            mul_out = FixedPoint(float(mul_out), signed=True, m=self.int_bits, n = self.frac_bits, rounding="down")
        return mul_out  
    
def addr_width(x):
    return math.ceil(math.log(x, 2)) 

# number of MAC-cores
N = 28
L = 10 # output layer

# Bit representation
int_bits, frac_bits = 6,7

# Creating MAC cores
MAC_array = []
for i in range(N):
    MAC_array.append(MAC(int_bits, frac_bits))
    
# Creating row-weights memory blocks
ROW_memory = []
for i in range(N):
    array_i = weights['L2DST.dense1.weight'][i].numpy()
    ROW_memory.append(MEM(len(array_i), int_bits, frac_bits))
    ROW_memory[i].initialization(array_i)
    
# Creating memory blocks for ROW biases
row_biases_array = weights['L2DST.dense1.bias']   
ROW_biases = MEM(len(row_biases_array), int_bits, frac_bits)
ROW_biases.initialization(row_biases_array)

COL_memory = []
for i in range(N):
    array_i = weights['L2DST.dense2.weight'][i].numpy()
    COL_memory.append(MEM(len(array_i), int_bits, frac_bits))
    COL_memory[i].initialization(array_i)
    
# Creating memory blocks for COL biases
col_biases_array = weights['L2DST.dense2.bias']   
COL_biases = MEM(len(col_biases_array), int_bits, frac_bits)
COL_biases.initialization(col_biases_array)

memory = []
for i in range(10):
    array_i = weights['W_o.weight'][i].numpy()
    memory.append(MEM(len(array_i), int_bits, frac_bits))
    memory[i].initialization(array_i)
    
# Creating memory blocks for COL biases
biases_array = weights['W_o.bias']   
biases = MEM(len(biases_array), int_bits, frac_bits)
biases.initialization(biases_array)

# Crearing internal RAM block for image storage
RAM_bock = MEM(N*N, int_bits, frac_bits)
RAM_bock.initialization(np.zeros(N*N))

# Createing tanh function module
Tanh = tanh(int_bits, frac_bits)

# Loading image to RAM block (stage 1)
for addr in range(N*N):
    RAM_bock.write_value(float(img_[addr]), addr)

# Row processing (stage 2)
for row in range(N):
    # MAC-core initialization
    for i in range(N):
        MAC_array[i].initialization(float(ROW_biases.mem[i]))
            
    for col in range(N):
        # Read data from memory
        in_data = RAM_bock.mem[row*N + col]
        
        for i in range(N):
            MAC_array[i].run(in_data, ROW_memory[i].mem[col], q_bit_to_remove=frac_bits)
            
    # Write result to memory
    for col in range(N):
        tmp = Tanh.calc(MAC_array[col].acc)
        RAM_bock.write_value(float(tmp), row*N + col) 

# Copy for visualization
FC1_out = RAM_bock.numpy()
        
# Column processing (stage 3)
for col in range(N):
    # MAC-core initialization
    for i in range(N):
        MAC_array[i].initialization(float(COL_biases.mem[i]))
        
    for row in range(N):
        # Read data from memory
        in_data = RAM_bock.mem[row*N + col]
        
        for i in range(N):
            MAC_array[i].run(in_data, COL_memory[i].mem[row], q_bit_to_remove=frac_bits)
         
    # Write result to memory
    for row in range(N):
        tmp = Tanh.calc(MAC_array[row].acc)
        RAM_bock.write_value(float(tmp), row*N + col) 

# Copy for visualization        
FC2_out = RAM_bock.numpy()

# Classification (stage 4)
final_result = []  # Softmax output

# Run MAC-cores
for i in range(L):  
    # MAC-core initialization
    MAC_array[i].initialization(float(biases.mem[i])) 

    # 784 MAC iteration
    for j in range(N * N):  
        MAC_array[i].run(RAM_bock.mem[j] , memory[i].mem[j], q_bit_to_remove=frac_bits)  

    # save FC output
    final_result.append(float(MAC_array[i].acc))

# Convert to Numpy
final_result = np.array(final_result)

# Output FC layer
print("Softmax input:", final_result)

# Result image
predicted_label = np.argmax(final_result)
print("Predicted label:", predicted_label)

FC2_out_ = FC2_out.reshape(N,-1)      
FC1_out_ = FC1_out.reshape(N,-1)

fig = plt.figure(figsize=(8,4))
plt.subplot(2,3,1)
plt.imshow(e1.detach().numpy())
plt.title('Float precision')
plt.colorbar()
plt.subplot(2,3,2)
plt.imshow(FC1_out_)
plt.title('Fixed point')
plt.colorbar()
plt.subplot(2,3,3)
plt.imshow(e1.detach().numpy() - FC1_out_) # , vmax=1, vmin=-1
plt.title('difference')
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(e2.detach().numpy())
plt.title('Float precision')
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(FC2_out_) 
plt.title('Fixed point')
plt.colorbar()
plt.subplot(2,3,6)
plt.imshow(e2.detach().numpy() - FC2_out_) # , vmax=1, vmin=-1
plt.title('difference')
plt.colorbar()

plt.subplots_adjust(wspace=0.4, hspace=0.4) 