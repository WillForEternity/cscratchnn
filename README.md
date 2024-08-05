# cscratchnn
A simple, proof of concept neural network in C, mainly for strengthening my mathematical intuition. 

Having just taken a digital logic circuit design class for the second time (and falling in love with it this time), I wanted to test the classic XOR function, known for its very simple nonlinear structure. When graphed, you'll see that it isn't linearly separable.

![cscratchnn](XORnonlinear.png)

I'd used numpy, keras, and pytorch to implement the simple structure of basic feedforward neural networks. I wanted to get even closer to the boilerplate and utilize C's speed. I want to learn how to use it for more complicated networks going forward. (This version only uses sigmoid activations and rank 1 tensors (arrays) for simplicity's sake)

## How To Run

This one is simple. No packages, no libraries. Just create a directory to store the file, and name your file something like `neuralnetwork.c`. Paste in my code.

To compile, cd into the directory, and: 

```bash
gcc -o neuralnetwork neuralnetwork.c -lm
```

To run: 

```bash
./neuralnetwork
```

##your results should look something like this:

![cscratchnn](Output.png)

the outputs that come from either 0 and 0 or 1 and 1 are closer to 0, whereas the inputs 1 and 0 or 0 and 1 produce results closer to 1. It's working. 
