import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

# data loading
with gzip.open('./data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

train_x, train_y = train_set

# parameters
_, NI = train_x.shape
L = 2
NL = 16
NO = 10

number_of_params = NI*NL + NL + (L-1)*NL**2 + (L-1)*NL + NL*NO + NO

print(f"Number of parameters: {number_of_params}")

itermax = 1500

learning_rate = 5

N = []

for i in range(L+2):
    if i == 0:
        N.append(NI)
    elif i == L+1:
        N.append(NO)
    else:
        N.append(NL)

# initialization of training parameters
weights = []
biases = []

for i in range(L+1):
    weights.append(np.random.uniform(low=-1.0, high=1.0, size=(N[i+1],N[i])))
    biases.append(np.random.uniform(low=-1.0, high=1.0, size=(N[i+1])))

# functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def calculateazKs(WK,aKm1s,bK):
    zKs = WK@aKm1s + bK
    aKs = sigmoid(zKs)
    return aKs, zKs

def front_propagation(a0s,weights,biases):
    aKm1s = [a0s]
    argss = []
    for i in range(L+1):
        aKs,zKs = calculateazKs(weights[i],aKm1s[i],biases[i])
        aKm1s.append(aKs)
        argss.append(zKs)
    return aKm1s,argss

def back_propagation(Delta,neurons,args,weights,d):
    eta = []
    for s in range(d):
        argss = args[s]
        deltas = Delta[s]
    
        etaLp1s = 2/d*deltas*dsigmoid(argss[-1])

        etasreversed = [etaLp1s]
        
        for K in range(L):
            etasreversed.append((weights[L-K].T@etasreversed[K])*dsigmoid(argss[L-1-K]))

        etas = etasreversed[::-1]
        eta.append(etas)

    gradbKL = []
    for K in range(L+1):
        sum = 0
        for s in range(d):
            sum += eta[s][K]
        gradbKL.append(sum)
    
    gradWKL = []
    for K in range(L+1):
        sum = 0
        for s in range(d):
            sum += np.outer(eta[s][K],neurons[s][K])
        gradWKL.append(sum)

    return gradbKL, gradWKL

def step(gradbKL, gradWKL, weights, biases):
    for K in range(L+1):
        weights[K] -= learning_rate*gradWKL[K]
        biases[K] -= learning_rate*gradbKL[K]
    return weights, biases

def training_loop(data,weights,biases):
    data_x, data_y = data
    d,_ = data_x.shape
    args = []
    neurons = []
    Delta = []
    for s in range(d):
        neuronss, argss = front_propagation(data_x[s],weights,biases)
        args.append(argss)
        neurons.append(neuronss)
        output = neuronss[-1]
        GT = []
        for i in range(NO):
            if i == data_y[s]:
                GT.append(1.0)
            else:
                GT.append(0.0)
        deltas = output-GT
        Delta.append(deltas)
    Loss = 0
    for i in range(len(Delta)):
        Loss += 1/d*np.linalg.norm(Delta[i])**2
    print(f"Training loss: {Loss}")
    gradbKL, gradWKL = back_propagation(Delta,neurons,args,weights,d)

    return Loss, gradbKL, gradWKL

def validation_loop(data,weights,biases):
    data_x, data_y = data
    d,_ = data_x.shape
    Delta = []
    success = 0
    for s in range(d):
        neuronss, argss = front_propagation(data_x[s],weights,biases)
        output = neuronss[-1]
        if np.argmax(output) == data_y[s]:
            success += 1
        GT = []
        for i in range(NO):
            if i == data_y[s]:
                GT.append(1.0)
            else:
                GT.append(0.0)
        deltas = output-GT
        Delta.append(deltas)
    Loss = 0
    for i in range(len(Delta)):
        Loss += 1/d*np.linalg.norm(Delta[i])**2
    print(f"Validation loss: {Loss}")
    accuracy = success/d
    print(f"Accuracy: {accuracy}")
    return Loss, accuracy

#main loop
iter = 0
iterations = []
training_losses = []
validation_losses = []
accuracy = []

while(iter < itermax):

    print(iter)

    train_loss, gradbKL, gradWKL = training_loop(train_set,weights,biases)
    val_loss, val_accuracy = validation_loop(valid_set,weights,biases)

    print(f"Learning rate: {learning_rate}")

    iter += 1
    iterations.append(iter)
    validation_losses.append(val_loss)
    training_losses.append(train_loss)
    accuracy.append(val_accuracy)

    if iter == itermax or val_loss < 0.01:
        break

    weights, biases = step(gradbKL,gradWKL,weights,biases)

    if np.mod(iter,400) == 0:
        learning_rate = learning_rate/2

plt.plot(iterations,validation_losses,label="val loss")
plt.plot(iterations,training_losses,label="train loss")
plt.plot(iterations,accuracy,label="accuracy")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(iterations,accuracy)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()
