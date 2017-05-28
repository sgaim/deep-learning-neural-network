# deep-learning-neural-network-
A quick to use neural network library that lets users hit the ground running on deep learning projects in Javascript modeled after Keras.

Creating `Network` that will hold all layers and adding layers
```
net = new Network()
net.add(Dense(units=64, input_dim=100))
net.add(Vanilla(10))
net.add(Dense(3))
```


3 Different types of Layers
1. Flat (aka Dense)
   `Dense(output)`
2. Vanilla RNN
   `Vanilla(output)`


Future Works
1. Improve layer addition
2. Sanitize Input in layer addition
3. Track Performance
2. LSTM Layer
3. CNN Layer
