import numpy as np
from pynq import Overlay
from PIL import Image
import matplotlib.pyplot as plt

# connect network
platform = Overlay("design_1.bit")
neural_net = platform.m_nn_0

# Address for in and out ports
NEURAL_NET_IN_ADDR = 3 * 4
NEURAL_NET_IN_COUNT = 4 * 4
NEURAL_NET_OUT_ADDR = 5 * 4
NEURAL_NET_IN_ADDR_RESET = 32 * 4
NEURAL_NET_IN_ADDR_START = 7 * 4
NEURAL_NET_OUT_RDY = 6 * 4

def float_to_fixed(val, int_bits, frac_bits):
    total_bits = int_bits + frac_bits
    scale = 1 << frac_bits
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    fixed_val = int(round(val * scale))

    if fixed_val > max_val or fixed_val < min_val:
        raise OverflowError(f"Value {val} out of range for {total_bits}-bit fixed-point.")
    if fixed_val < 0:
        fixed_val = (1 << total_bits) + fixed_val  # Two's complement
    return fixed_val

# load image from MNIST 
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # read head
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Load labels from MNIST
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # read head
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Proceccing image
def predict(image):
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, 0)
    pixel_values = list(image.getdata())
    width, height = image.size
    pixel_val = [(pixel - 127.5) / 127.5 for pixel in pixel_values] 
    pixel_values = [pixel_val[i * width: (i + 1) * width] 
                    for i in range(height)]
    
    pixel_count = 0
    output = neural_net.read(NEURAL_NET_OUT_ADDR) # read answer
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, 1)
    # pixel counter for correct transfer of each pixels
    for i in range(height):
        for j in range(width):
            neural_net.write(NEURAL_NET_IN_ADDR_START, 0)
            neural_net.write(NEURAL_NET_IN_COUNT, pixel_count) # send counter 
             # send pixel in Q6.7
            q67_val = float_to_fixed(pixel_values[i][j], 6, 10)
            neural_net.write(NEURAL_NET_IN_ADDR, q67_val)
            pixel_count = pixel_count + 1
    
    for i in range(10000):
        output = neural_net.read(NEURAL_NET_OUT_RDY)
        if(output):
          break 
        
    output = neural_net.read(NEURAL_NET_OUT_ADDR)
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, 0)
    return output

# Empty confusion matrix
def create_confusion_matrix(num_classes):
    return np.zeros((num_classes, num_classes), dtype=int)

# Update confusion matrix
def update_confusion_matrix(conf_matrix, true_label, predicted_label):
    conf_matrix[true_label][predicted_label] += 1

num_classes = 10
conf_matrix = create_confusion_matrix(num_classes)

# path to MNIST
mnist_images_file = 't10k-images-idx3-ubyte'  
mnist_labels_file = 't10k-labels-idx1-ubyte'
images = load_mnist_images(mnist_images_file)
true_labels = load_mnist_labels(mnist_labels_file)
count = 0
# Test on FPGA and create a confusion matrix
for img_array, true_label in zip(images, true_labels):
      if (count % 1000 == 0):
        print(f'Conf Matrix for {count} img')
        print(conf_matrix)
      
      img = Image.fromarray(img_array) 
      prediction = predict(img)
      print(f"{count}: {true_label} | {prediction}")

      update_confusion_matrix(conf_matrix, true_label, prediction)
      count += 1

print("Confusion matrix:")
print(conf_matrix)