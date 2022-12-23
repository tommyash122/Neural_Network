# Neural_Network - Iris flower data set

## About
A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain.    
It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain. 

The Iris flower data set or Fisher's Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his
1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.    
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).    
Four features were measured from each sample:
the length and the width of the sepals and petals, in centimeters.    
Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.    
Fisher's paper was published in the Annals of Eugenics and includes discussion of the contained techniques' applications to the field of phrenology.    
Link for more info on wikipedia:    
https://en.wikipedia.org/wiki/Iris_flower_data_set    

Link to a short video explaining the running of the Neural Network, viewing the outputs differences and explanation of how I get the data.(Hebrew)    
https://youtu.be/_gGUMTpsKPc    


## System requirements 
1) Python 3.4 and above installed.
2) files main.py and Neural_Network.py needs to be on the same folder.

Trainig process pic:    
![Capture](https://user-images.githubusercontent.com/84855441/209366407-f99b77c0-6f02-488e-b6e7-069df78b3b6a.PNG)    

## Explanation    
Our neural network will look like this:    
![NNPic](https://user-images.githubusercontent.com/84855441/209366843-a4dd3dc3-8466-4fdb-9697-72b09483d640.PNG)    
Our program is an interactive neural network that during its oparation learns from the database provided to it and improves its ability to answer based on that information.    
After the learning process is finished, a user interface simulating a command line will open where the user can enter inputs as he wishes into the network and this as a response will generate appropriate answers based on its past learning.    
At any time the user can end the program by typing `exit`    

Our database on which we will rely Iris flower data set which we will obtain by importing the **sklearn** library.    
In addition we will use the **numpy** library in order to perform the many matrices operations that will happen in this network.

### The network architecture    
• **The input layer** will have 4 nodes, as the number of attributes that must be entered for each iris type flower that we have.
  Into this layer will be inserted the input which, as mentioned, is the attributes of each iris flower found in the given iris data set.    
  The four nodes will receive the following values respectively from left to right:    
  sepal length, sepal width, petal length, petal width.    
• **One intermediate layer (Hidden layer)** that will contain 5 nodes.    
  Each node hi in this layer will receive the value of the inner product of the values of the nodes from the input layer with the values of the weights of the edges entering hi.    

![weights_input2hidden_example_values](https://user-images.githubusercontent.com/84855441/209371840-e6d7a3f5-9a99-4cc6-a0e6-e948c86de2a8.jpg)    
    
    
• **The output layer** that will contain 3 nodes as the number of classes to be classified.    
The output layer will produce the output that the neural network infers, based on the data it currently has, which is the correct class for the features of the input fed to it.    
• Every node from layer x is connected to every node in the next layer x+1 except for the nodes in the output layer, the edges that connect the nodes will be called the means of this connection and these will be initialized for the first time with random values ("weights") in the value range [1,-1].
