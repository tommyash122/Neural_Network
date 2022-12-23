# Neural_Network - Iris flower data set

![gif](https://149695847.v2.pressablecdn.com/wp-content/uploads/2022/01/ezgif.com-gif-maker-17.gif)


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

Check out some data [graphs](##Let's-look-at-some-graphs)    

## System requirements 
1) Python 3.4 and above installed.
2) files main.py and Neural_Network.py needs to be on the same folder.

## Explanation    
Our neural network will look like this:    
![NNPic](https://user-images.githubusercontent.com/84855441/209366843-a4dd3dc3-8466-4fdb-9697-72b09483d640.PNG)    
Our program is an interactive neural network that during its oparation learns from the database provided to it and improves its ability to answer based on that information.    
After the learning process is finished, a user interface simulating a command line will open where the user can enter inputs as he wishes into the network and this as a response will generate appropriate answers based on its past learning.    
At any time the user can end the program by typing `exit`    

Our database on which we will rely Iris flower data set which we will obtain by importing the **sklearn** library.    
In addition we will use the **numpy** library in order to perform the many matrices operations that will happen in this network.
There is a linear trend line made up of red dots indicating the decreasing trend of the error rate during training which indicates that the network is improving in its detection ability over the period.

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
• **Our activation function** will be the sigmoid function and we will also use its derivative to calculate the error function.    
![image](https://user-images.githubusercontent.com/84855441/209396895-c6e72684-83fb-4a02-ba77-7523c50f64fc.png)    
![image](https://user-images.githubusercontent.com/84855441/209396914-5fa9aa66-f22d-4446-94d8-5f9681fd67a9.png)    
    
    
### Parameters of the optimization process:    
• Our system training approach will be the gradient descent method, the error calculation will be done by subtracting the output values from the value of the correct answer.    
We multiply the obtained value by the sigmoid derivative function at the point in order to reduce the value of the error function.    
We will multiply the value obtained by the learning rate coefficient chosen to be 0.15.
• The size of our batch will be 80% of the size of the total input and is 150*0.8 = 120, that is, in each Epoch we will train the network using 120 inputs, so that in each round they will chosen randomly, and we will test the network with the remaining 30 inputs.    
In each such round, the training and testing groups are randomly selected from the entire database in our possession.    
• The size of the Epochs variable = 200, which means that we will run the entire data group 200 times, as mentioned above, each time our batch size will be 120.    
• As mentioned above, the learning rate will be 0.15.    

### Output while running the system for training:
![image](https://user-images.githubusercontent.com/84855441/209398379-cc33e550-b318-45bd-bd42-64ea490a54ae.png)    
You can clearly see the increase in the degree of accuracy (train, test) of the system throughout the running of the epochs.    
**Epoch 1 (train, test) = (35.83%, 23.33%)**    
...    
**Epoch 10 (train, test) = (85.00%, 86.67%)**    
...    
**Epoch 200 (train, test) = (95.00%, 96.67%) ** 
    
Hence our degree of accuracy will remain, according to this run according to the Epoch data **200**, with a degree of accuracy on new data of **96.67%**.    
We will also notice that from epoch **10** to **200** the degree of accuracy rises and falls but remains stable around **95%**.    
It can be concluded from this that the learning rate of the network decreases as a function of the number of epochs that were used to train it with, but I will note in addition that it can be seen that the pattern is still an upward pattern learning despite the slowdown in the learning rate, we will present this later.    
I will mention here that it is sometimes possible to reach a level of accuracy of **100%** within these **200** epochs as we will see in epoch **40** for our test group.
In general, I will point out that these data change from time to time and there will be situations where we will start in the first epoch with success data of **60%** and the tenth epoch will only rise to **70%**.

### UI :    
![image](https://user-images.githubusercontent.com/84855441/209399122-a4c955bf-dc0c-4bc6-ae31-ec1a84b7571c.png)    
As I mentioned above, in this system it is possible to have a UI (user interface).    
The user can communicate with the system by enter input (data of some iris flower that he wants to check about) and the system will return an answer.
setosa, versiclolor, virginica respectively and That user can verify and concluded that the system has learned their properties well.  
In addition, it seems that the fourth input does not exactly match any flower from our database because there is no flower in the database whose petal width is **2.6**, there are other features that have a value of **2.6** and but as mentioned there is no petal width in the data with that value, in addition we will notice that all the parameters entered at the forth input is slightly different from the third input and yet the system has chosen to label this data as a virginica flower because, based on the knowledge and the learning process it has undergone, it concludes that these data are the closest approximation to those of virginica and the highest probability that a flower of this type is the one that will have these characteristics.

## Let's look at some graphs
### Network training error rate across epochs    
![image](https://user-images.githubusercontent.com/84855441/209400115-436f1e62-45f3-4a2c-b3f5-6522c6071f2a.png)    
It can be seen that the error rates for the training throughout the periods starts high compared to the rest, it is about **0.3** in the first epoch but immediately after that it drops significantly to the 0.05 area and from there it rises and falls but remains at an average of **0.053**.    
There is a linear trend line made up of red dots indicating the decreasing trend of the error rate during training which indicates that the network is improving in its detection ability over the period.    
#### Training set statistics:    
Average 0.053, median 0.041, minimum value 0, maximum value 0.341.    

### Network testing error rate across epochs
![image](https://user-images.githubusercontent.com/84855441/209400356-282afc2e-045b-449a-83e8-3cf440d0d9e3.png)    
It can be noticed that the data of the test group is slightly different from the data of the training group, because in the test group the data is not for use in     training the system, that is, the backpropagation algorithm which tune the weights of the edges to the right values is not performed on this data,    
only forward-propagation whose role is to feed as an input to the network and generate output from it.    
Hence this group is used for "new" data that the system was not intended for and their role is to really test the degree of learning.    
There is a linear trend line made up of red dots indicating the decreasing trend of the error rate during training which indicates that the network is improving in its detection ability over the period.    
#### Testing set statistics:    
Average 0.053, median 0.033, minimum value 0, maximum value 0.433.    

### Network error rate across epochs    
![image](https://user-images.githubusercontent.com/84855441/209400987-c637f7ba-580a-4e98-8ec4-994890dc4495.png)    
I have set up a chart here showing both graphs.
The blue represents the degree of error of the training group,    
The dimmed orange represents the degree of error of the test group.    
It can be proven that the degree of error in the examination group is slightly higher than that of the training group, which as stated above is "new" data for the system and its weights were not adjusted according to them but according to the training group, but it can still be seen that the last degree of error and hence the current one by which the system is measured is **0.016** for the training and **0.033** for the test and hence we conclude that for each new input we enter for our system there is a degree of error of **0.033** which means **96.67%** that the system will provide a correct answer.    
    
### Overfitting check    
![image](https://user-images.githubusercontent.com/84855441/209401478-b5238a90-a06c-406b-a7fd-c101183a0551.png)
Here I created a copy of the previous chart called Network error rate across epochs.
As mentioned this diagram visually shows the two graphs of the training and the test as a function of epochs.    
For this chart I dimmed the graphs themselves in the chart and created trendlines for each graph.    
The orange trend line represents the graph of the test    
The blue trend line represents the training graph.    
It can be clearly seen that there is a downward trend in the level of error throughout the measurement period and it can even be seen that the degree of error in the test group falls below the degree of error in the training group from epoch 73 or so, meaning in other words the system independently identifies inputs that it does not recognize with a degree of error better than that on inputs that it does recognize.    
I will point out that the degree of this difference is negligible, which means the difference is not particularly large, which indicates that the system learned the data well and we did not observe a case of overfitting here, as as we know, there is some point where the system becomes too dependent on the training inputs and does not know how to recognize other inputs independently, which is not our case. :)    


![image](https://user-images.githubusercontent.com/84855441/209402058-1622dab0-3ddb-4562-976a-8c6db0746230.png)
    




