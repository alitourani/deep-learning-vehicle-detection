# deep-learning-vehicle-detection
(Vehicle Detection based on Faster R-CNN)

Faster R-CNN has two vital modules including a deep fully convolutional network called Region Proposal Network (RPN) and a Fast R-CNN network to use the region proposals. Hereby, the RPN provides information about the object for the Fast R-CNN module. The main contributions of this work includes generating a Faster R-CNN model for vehicle detection, training the model by a standard vehicle image dataset and evaluating the final results on real condition data.

# Vehicle Detection Process
To generate the model for vehicle detection, the structure of hidden layers, the number of neurons in each layer and the optimized weight of each neuron should be clearly specified. In order to simplify the tough process of training the DNN for vehicle detection, a residual learning framework known as “ResNet-50” has been used in this work.

Furthur information about this approach can be found in below paper:
A. Tourani, S. Soroori, A. Shahbahrami, S. Khazaee and A. Akoushideh, "A Robust Vehicle Detection Approach based on Faster R-CNN Algorithm," The 4th International Conference on Pattern Recognition and Image Analysis, Tehran, Iran, March 2019.
