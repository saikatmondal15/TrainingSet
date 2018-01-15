package skynet;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import org.encog.engine.network.activation.*;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;

/**
* A parent class for other OutputWriters. This class holds all of the

* shared methods required to create a file and output the code/formula for
* a trained Artificial Neural Network.
* @author bwinrich
*/
public abstract class OutputWriter {

/**
* The file to write to
*/
protected File file2;

/**
* A BufferedWriter for file writing
*/
protected BufferedWriter bw2 = null;


/**
* The name of the file
*/
protected String outputName;

/**
* Does the data set have headers?
*/
protected boolean hasHeaders;

/**
* Array to hold the column names (only if the data set has headers)
*/
protected String[] columnNames;

/**
* The data structure for the ANN
*/
protected BasicNetwork network;

/**
* The number of input nodes
*/
protected int inputCount;

/**
* The number of output nodes
*/
protected int outputCount;


/**
* The number of neurons in each layer (including bias neurons)
*/
protected int[] numberOfTotalNeurons;

/**
* The number of neurons in each layer (excluding bias neurons)
*/
protected int[] numberOfNormalNeurons;

/**
* The value of the biases of each layer, if applicable
*/
protected double[] biases;

/**
* The flattened version of the ANN
*/
protected FlatNetwork myFlat;

/**
* The number of layers in the ANN
*/
protected int layers;


/**
* Default constructor
*/
public OutputWriter(){}

/**
* Mutator method for hasHeaders
* @param hasHeaders the hasHeaders to set
*/
public void setHasHeaders(boolean hasHeaders) {
this.hasHeaders = hasHeaders;
}

/**
* Mutator method for columnNames
* @param columnNames the columnNames to set
*/
public void setColumnNames(String[] columnNames) {
this.columnNames = columnNames;
}


/**
* Mutator method for network
* @param network the network to set
*/
public void setNetwork(BasicNetwork network) {
this.network = network;
}

/**
* Mutator method for inputCount
* @param inputCount the inputCount to set
*/
public void setInputCount(int inputCount) {
this.inputCount = inputCount;
}

/**
* Mutator method for outputCount
* @param outputCount the outputCount to set
*/
public void setOutputCount(int outputCount) {
this.outputCount = outputCount;
}

/**
* Mutator method for numberOfTotalNeurons
* @param numberOfTotalNeurons the numberOfTotalNeurons to set
*/
public void setNumberOfTotalNeurons(int[] numberOfTotalNeurons) {
this.numberOfTotalNeurons = numberOfTotalNeurons;
}

/**
* Mutator method for numberOfNormalNeurons
* @param numberOfNormalNeurons the numberOfNormalNeurons to set
*/
public void setNumberOfNormalNeurons(int[] numberOfNormalNeurons) {
this.numberOfNormalNeurons = numberOfNormalNeurons;
}

/**
* Mutator method for biases
* @param biases the biases to set
*/
private void setBiases(double[] biases) {

this.biases = biases;
}

/**
* Mutator method for myFlat
* @param myFlat the myFlat to set
*/
private void setMyFlat(FlatNetwork myFlat) {
this.myFlat = myFlat;
}

/**
* Mutator method for layers
* @param layers the layers to set
*/
public void setLayers(int layers) {
this.layers = layers;
}

/**
* Some variables can be initialized using information already passed to
* the class.
*/
public void initializeOtherVariables(){
setMyFlat(network.getStructure().getFlat());
setBiases(myFlat.getBiasActivation());

}

/**
* Creates the file used for output.
* @param output2Name the name of
*/
public abstract void createFile(String output2Name);

/**
* Writes to the output file. Each String passed as a parameter is
* written on its own line
* @param stuff The line to be written to the file
*/
protected void writeTwo(String stuff){
try{
bw2.write(stuff);
bw2.newLine();
}catch (IOException e){
// TODO Auto-generated catch block

e.printStackTrace();
}
}

/**
* Parses the equation of the activation function and returns it in
* String form
* @param af The activation function to parse
* @param varName The variable passed to the activation function
* @param targetVarName The variable the result of the activation
* function will be stored in
* @return The parsed form of the activation function in String form
*/
protected abstract String parseActivationFunction(ActivationFunction af,
String varName, String targetVarName);

/**
* A lengthy method for writing the code/formula for the neural network
* to a file. Each child class will have its own implementation.
*/
public abstract void writeFile();
}
