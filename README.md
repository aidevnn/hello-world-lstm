# hello-world-lstm
First LSTM

The base text for training is simply 3 times repeated "Hello World\n"

The input layer contains X_size = 9 nodes for unique characters of the sequence.

The hidden layer contains H_size = 18 nodes.
 
And recurrence time steps T_steps = 12 steps for the length of initial word

### Output

```
X_size:9 H_size:18 T_steps:12

Epochs:     0/4000 Loss:26.368168 Time:    20 ms
HHdHdldHedddo eoeHeWdHd
oWHleldloW

...

Epochs:  2000/4000 Loss:3.630082 Time:  2450 ms
Hello World
HellorWorld
Hello World

...

Epochs:  4000/4000 Loss:0.494055 Time:  4762 ms
Hello World
Hello World
Hello World

```