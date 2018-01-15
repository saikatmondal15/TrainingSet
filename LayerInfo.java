
package skynet;
/**
* A simple class, designed to hold the information required to create a
* layer in the neural network
* @author bwinrich
*/


class LayerInfo{

/**
* An integer for the type of activation function (see comments in
* config file for details)
*/
private int activationFunction;

/**
* A boolean for if the layer has a bias node or not
*/
private boolean isBiased;

/**
* An integer for the number of normal neurons in the layer
*/
private int neurons;

/**
* A constructor with parameters. We have no need for a default
* constructor
* @param activationFunction type of activation function
* @param isBiased is there a bias node
* @param neurons number of normal neurons
*/
public LayerInfo(int activationFunction, boolean isBiased, int neurons){
this.activationFunction = activationFunction;
this.isBiased = isBiased;
this.neurons = neurons;
}

/**
* Accessor method for activationFunction
* @return the activationFunction
*/
public int getActivationFunction() {
return activationFunction;
}

/**
* Accessor method for isBiased
* @return the isBiased
*/
public boolean isBiased() {
return isBiased;

}

/**
* Accessor method for neurons
* @return the neurons
*/
public int getNeurons() {
return neurons;
}

/** A method used for returning the information for the layer in an
* easy-to-read format, so that it can be printed.
* @see java.lang.Object#toString()
*/
@Override
public String toString(){

String activation = null;

switch(activationFunction){
case -1:
activation = "n/a";
break;
case 0:
activation = "Sigmoid";
break;
case 1:
activation = "Hyperbolic Tangent";
break;
case 2:
activation = "Linear";
break;
case 3:
activation = "Elliott";
break;
case 4:
activation = "Gaussian";
break;
case 5:
activation = "Logarithmic";
break;
case 6:
activation = "Ramp";
break;
case 7:
activation = "Sine";

break;
case 8:
activation = "Step";
break;
case 9:
activation = "BiPolar";
break;
case 10:
activation = "Bipolar Sigmoid";
break;
case 11:
activation = "Clipped Linear";
break;
case 12:
activation = "Competitive";
break;
case 13:
activation = "Elliott Symmetric";
break;
case 14:
activation = "Softmax";
break;
case 15:
activation = "Steepened Sigmoid";
break;
default:
activation = "Invalid";
break;
}

return ("Layer: (" + activation + ", " + isBiased + ", " + neurons
+ ")");
}
}
