# Model
softmax(AX + B) = X with random noise added after several applications of gradient descent. This is one cycle of the model. The random noise is added to simulate the quantum vacuum. This model is an auto-encoder which computes both page rank (X) and word vectors (A). (A) is also an adjacency matrix describing the connections between space (or is it particles?). Column vectors of X describe the probability of finding a particle in a particular place. Column vectors sum to 1. I should probably re-implement this model using complex numbers... 8 particle simulation at the bottom of the page is most interesting.

# Simulations

## 1 particle universe simulation
![simulation](verse_1.gif?raw=true)

## 2 particle universe simulation
![simulation](verse_2.gif?raw=true)

## 3 particle universe simulation
![simulation](verse_3.gif?raw=true)

## 8 particle universe simulation
![simulation](verse_8.gif?raw=true)

## 8 particle universe simulation with low quantum vacuum
![simulation](verse_blackhole_8.gif?raw=true)
