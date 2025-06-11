from pynq import Overlay 
from PIL import Image 
from time import sleep
import matplotlib.pyplot as plt

# connect network
platform = Overlay("design_1.bit")
neural_net = platform.m_nn_0

# Address for in and out ports
NEURAL_NET_IN_ADDR = 3 * 4
NEURAL_NET_IN_COUNT = 4 * 4
NEURAL_NET_OUT_ADDR = 5 * 4
NEURAL_NET_IN_ADDR_RESET = 6 * 4

image_list = ["pic5.png", "pic9.png", "pic4.png", "pic1.png", "pic8.png"]
for file_name in image_list:
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, int(0))
    image = Image.open(file_name)
    pixel_values = list (image.getdata())
    width, height = image.size
    pixel_val = [(pixel - 127.5) / 127.5 for pixel in pixel_values] 
    pixel_values = [pixel_val[i * width: (i + 1) * width] 
                    for i in range(height)]
    
    pixel_count = 1
    output = neural_net.read(NEURAL_NET_OUT_ADDR) 
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, 1)
    output = neural_net.read(NEURAL_NET_OUT_ADDR)
    for i in range (height):
        for j in range(width):
            neural_net.write(NEURAL_NET_IN_COUNT, pixel_count) 
             # send pixel in Q12.12
            neural_net.write(NEURAL_NET_IN_ADDR, int(pixel_values[i][j] * 2**12))
            pixel_count = pixel_count + 1


    output = neural_net.read(NEURAL_NET_OUT_ADDR)
    plt.imshow(image)
    plt.title(f'NN output = {output}')
    plt.colorbar()
    plt.show()
    print (output)
    neural_net.write(NEURAL_NET_IN_ADDR_RESET, int(0))