package skynet;

import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.*;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.TrainingSetUtil;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
/**
* The main class for the program. The purpose of this program is the train
* an Artificial Neural Network and output source code for it, so that it
* can be used in other projects.
* @author bwinrich
*
*/
public class NeuralGenerator {
/**
* A BufferedWriter for our error output
*/
private BufferedWriter bw1 = null;
/**
* An OutputWriter for our code output
*/
private OutputWriter myOutput;
/**
* The number of neurons in each layer (including bias neurons)
*/
private int[] numberOfTotalNeurons;
/**
* The number of neurons in each layer (excluding bias neurons)
*/
private int[] numberOfNormalNeurons;
/**
* The location of the .csv file for the training data set
*/
private String filePath = null;
/**
* Does the data set have headers?
*/
private boolean hasHeaders = false;
/**
* The number of input nodes
*/
private int numOfInput = 0;
/**
* The number of output nodes
*/
private int numOfOutput = 0;
/**
* The number of hidden layers in the network
*/
private int numOfHiddenLayers = 0;
/**
* An array holding the information for each layer (activation function,
* bias, number of neurons)
*/
private LayerInfo[] allMyLayers = null;
/**
* The network will train until the error is this value or lower
*/
private double desiredTrainingError = 0.01;
/**
* The maximum number of epochs the network will train
*/
private int numOfEpochs = 0;
/**
* The learning rate for the network (backpropagation only)
*/
private double learningRate = 0;
/**
* The momentum for the network (backpropagation only)
*/
private double momentum = 0;
/**
* The type of file the error output will be (0: .txt, 1: .csv)
*/
private int output1Type = 0;
/**
* The type of file the code output will be (0: .txt, 1/2: .java)
*/
private int output2Type = 0;
/**
* The name of the error output file
*/
private String output1Name = null;
/**
* The name of the code output file
*/
private String output2Name = null;
/**
* The data structure for the ANN
*/
private BasicNetwork network = null;
/**
* Array to hold the column names (only if the data set has headers)
*/
private String[] columnNames;
/**
* The type of network the ANN will be (0: Resilient propagation,
* 1: backpropagation)
*/
private int networkType;
/**
* The training data set
*/
private MLDataSet trainingSet;
/**
* The main method.
* @param args Should contain the location of the config file
*/
@SuppressWarnings("unused")
public static void main(final String args[]) {
if (args.length == 0){
System.out.println("Error: No file");
}else{
String configFilePath = args[0];
NeuralGenerator myThesis = new NeuralGenerator(configFilePath);
}
}
/**
* Constructor for the class.
* @param configFilepath The location of the config file
*/
public NeuralGenerator(String configFilepath){
newMain(configFilepath);
}
/**
* The driver method, which will call all other necessary methods
* required for the execution of the program
* @param configFilePath The location of the config file
*/
private void newMain(String configFilePath){
//Import the config file, and the necessary information
validateConfig(configFilePath);
//Create the first output file
System.out.println("Initializing first output file...");
initializeOutput1();
// create a neural network
System.out.println("Creating network...");
createNetwork();
//Import data set
System.out.println("Importing csv file...");
trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH,
filePath, hasHeaders, numOfInput, numOfOutput);
//Just because I prefer working with arrays instead of arrayLists
if(hasHeaders){
List<String> columns = TrainingSetUtil.getColumnNames();
int width = columns.size();
columnNames = new String[width];
for (int i = 0; i < width; i++ ){
columnNames[i] = columns.get(i);
}
}
// train the neural network
train();
// Close the first file after we’re done with it
try{
if(bw1!=null)
bw1.close();
}catch(Exception ex){
System.out.println("Error in closing the BufferedWriter"+ex);
}
// test the neural network
System.out.println("");
System.out.println("Neural Network Results:");
for(MLDataPair pair: trainingSet ) {
final MLData output = network.compute(pair.getInput());
String printInput = "";
String printOutput = "";
String printIdeal = "";
//Iterate through the input data to generate a string.
for(int i = 0; i < numOfInput; i++){
printInput += pair.getInput().getData(i);
printInput += ",";
}
//We do the same thing with the output data
for(int i = 0; i < numOfOutput; i++){
printOutput += output.getData(i);
printOutput += ",";
}
//And the same thing for the ideal data
for(int i = 0; i < numOfOutput; i++){
printIdeal += pair.getIdeal().getData(i);
if((i+1) != numOfOutput){
printIdeal += ",";
}
}
System.out.println(printInput + " actual=" + printOutput
+ " Ideal=" + printIdeal);
}
//Some additional numbers that we need
int layers = network.getLayerCount();
numberOfTotalNeurons = new int [layers];
numberOfNormalNeurons = new int [layers];
for (int i = 0; i<layers; i++)
{
numberOfTotalNeurons[i] = network.getLayerTotalNeuronCount(i);
numberOfNormalNeurons[i] = network.getLayerNeuronCount(i);
}
System.out.println("\n");
//Initialize the OutputWriter
System.out.println("Initializing Second Output File...");
initializeOutput2();
System.out.println("Writing to file...");
myOutput.writeFile();
System.out.println("Done.");
Encog.getInstance().shutdown();
}
/**
* This method handles the training of the Artificial Neural Network
*/
private void train() {
Propagation train = null;
//Different networks will be created based on the type listed in the
//config file
switch(networkType){
case 0:
train = new ResilientPropagation(network, trainingSet);
break;
case 1:
train = new Backpropagation(network, trainingSet, learningRate,
momentum);
break;
default:
break;
}
int epoch = 1;
System.out.println("");
System.out.println("Training...");
System.out.println("");
//Training the network
do {
train.iteration();
//We write the error to the first output file
writeOne(epoch, train.getError());
epoch++;
} while((train.getError() > desiredTrainingError)
&& (epoch < numOfEpochs));
//Training will continue until the error is not above the desired
//error, or until the maximum number of epochs has been reached
train.finishTraining();
}
/**
* Helped method for creating the ANN and adding layers to it
*/
private void createNetwork() {
network = new BasicNetwork();
for (LayerInfo myLayer:allMyLayers)
{
switch (myLayer.getActivationFunction()){
case -1: //The input layer doesn’t have an activation function
network.addLayer(new BasicLayer(null, myLayer.isBiased(),
myLayer.getNeurons()));
break;
case 0:
network.addLayer(new BasicLayer(new ActivationSigmoid(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 1:
network.addLayer(new BasicLayer(new ActivationTANH(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 2:
network.addLayer(new BasicLayer(new ActivationLinear(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 3:
network.addLayer(new BasicLayer(new ActivationElliott(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 4:
network.addLayer(new BasicLayer(new ActivationGaussian(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 5:
network.addLayer(new BasicLayer(new ActivationLOG(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 6:
network.addLayer(new BasicLayer(new ActivationRamp(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 7:
network.addLayer(new BasicLayer(new ActivationSIN(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 8:
network.addLayer(new BasicLayer(new ActivationStep(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 9:
network.addLayer(new BasicLayer(new ActivationBiPolar(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 10:
network.addLayer(new BasicLayer(
new ActivationBipolarSteepenedSigmoid(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 11:
network.addLayer(new BasicLayer(new ActivationClippedLinear(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 12:
network.addLayer(new BasicLayer(new ActivationElliottSymmetric(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
case 13:
network.addLayer(new BasicLayer(new ActivationSteepenedSigmoid(),
myLayer.isBiased(), myLayer.getNeurons()));
break;
default:
//Unimplemented activation function: Softmax (complicated)
//Unimplemented activation function: Competitive
//(non-differentiable)
System.out.println("Error: This activation function is "
+ "either invalid or not yet implemented");
break;
}
}
network.getStructure().finalizeStructure();
network.reset();
}
/**
* This method creates the error output file
*/
private void initializeOutput1() {
String output1NameFull = null;
//File type is specified in the config file
switch(output1Type){
case 0:
output1NameFull = output1Name + ".txt";
break;
case 1:
output1NameFull = output1Name + ".csv";
break;
default:
//More cases can be added at a later point in time
System.out.println("Invalid output 1 type");
}
try{
File file1 = new File(output1NameFull);
if (!file1.exists()) {
file1.createNewFile();
}
FileWriter fw1 = new FileWriter(file1);
bw1 = new BufferedWriter(fw1);
//Header line for a .csv file
if (output1Type == 1){
bw1.write("Epoch,Error");
bw1.newLine();
}
}catch (IOException e){
// TODO Auto-generated catch block
e.printStackTrace();
}
}
/**
* This method creates the code output file
*/
private void initializeOutput2(){
//File type is specified in the config file
switch(output2Type){
case 0:
myOutput = new OutputWriterTxt();
break;
case 1:
myOutput = new OutputWriterJava(true);
break;
case 2:
myOutput = new OutputWriterJava(false);
break;
default:
//More cases can be added if additional classes are designed
System.out.println("Invalid output 2 type");
break;
}
//Creating the file
myOutput.createFile(output2Name);
//Passing all of the necessary network information to the OutputWriter
myOutput.setNetwork(network);
myOutput.setInputCount(numOfInput);
myOutput.setOutputCount(numOfOutput);
myOutput.setLayers(numOfHiddenLayers+2);
myOutput.setNumberOfTotalNeurons(numberOfTotalNeurons);
myOutput.setNumberOfNormalNeurons(numberOfNormalNeurons);
myOutput.setHasHeaders(hasHeaders);
myOutput.setColumnNames(columnNames);
myOutput.initializeOtherVariables();
}
/**
* Helper method for writing to the error output file
* @param epoch Number of times the network has been trained
* @param error Training error for that epoch
*/
private void writeOne(int epoch, double error) {
String temp = null;
//Format depends on file type
switch(output1Type){
case 0:
temp = "Epoch #" + epoch + " Error:" + error;
break;
case 1:
temp = "" + epoch + "," + error;
break;
default:
temp = "Invalid output 2 type";
break;
}
//Output the error to the console before writing it to the file
System.out.println(temp);
try {
bw1.write(temp);
bw1.newLine();
} catch (IOException e) {
// TODO Auto-generated catch block
e.printStackTrace();
}
}
/**
* Helper method for retrieving lines from the config file. Comments are
* not considered valid lines.
* @param d The BufferedReader for the config file
* @return The next valid line from the config file
* @throws IOException
*/
private String nextValidLine(BufferedReader d) throws IOException
{
String validLine = null;
boolean isValid = false;
if (d.ready()){
do{
String str = d.readLine();
if (str.length() != 0){
//Eliminate extra space
str = str.trim();
//Comments start with %, and are not considered valid
if (str.charAt(0) != ’%’){
validLine = str;
isValid = true;
}
}
}while (!isValid && d.ready());
}
return validLine;
}
/**
* A lengthy method for validating the config file. All information from
* the config file is stored into data members so it can be accessed by
* other methods.
* @param configFilepath The location of the config file
*/
public void validateConfig(String configFilepath)
{
try{
File myFile = new File(configFilepath);
FileInputStream fis = null;
BufferedReader d = null;
fis = new FileInputStream(myFile);
d = new BufferedReader(new InputStreamReader(fis));
//First, we store the file path of the .csv file
if (d.ready()){
filePath = nextValidLine(d);
}
//Next, we store if the csv file has headers or not
if (d.ready()){
hasHeaders = Boolean.parseBoolean(nextValidLine(d));
}
//Next, we store the number of input parameters
if (d.ready()){
numOfInput = Integer.valueOf(nextValidLine(d));
}
//Next, we store the number of output parameters
if (d.ready()){
numOfOutput = Integer.valueOf(nextValidLine(d));
}
//Next, we store the number of hidden layers
if (d.ready()){
numOfHiddenLayers = Integer.valueOf(nextValidLine(d));
}
//Next, we store the information for our hidden layers
allMyLayers = new LayerInfo[numOfHiddenLayers+2];
String layer = null;
int activationFunction;
boolean isBiased;
int neurons;
for (int i = 1; i < numOfHiddenLayers+1; i++){
if (d.ready()){
layer = nextValidLine(d);
layer = layer.trim().toLowerCase();
layer = layer.substring(1,layer.length()-1);
String[] layers = layer.split(",");
for (String l:layers){
l = l.trim();
}
activationFunction = Integer.valueOf(layers[0].trim());
isBiased = Boolean.parseBoolean(layers[1].trim());
neurons = Integer.valueOf(layers[2].trim());
allMyLayers[i] =
new LayerInfo(activationFunction, isBiased, neurons);
}
}
//Next, we store the information for the input layer
if (d.ready()){
layer = nextValidLine(d);
layer = layer.trim().toLowerCase();
layer = layer.substring(1,layer.length()-1);
isBiased = Boolean.parseBoolean(layer.trim());
allMyLayers[0] = new LayerInfo(-1, isBiased, numOfInput);
}
//Finally, we store the information for the output layer
if (d.ready()){
layer = nextValidLine(d);
layer = layer.trim().toLowerCase();
layer = layer.substring(1,layer.length()-1);
String[] layers = layer.split(",");
activationFunction = Integer.valueOf(layers[0].trim());
allMyLayers[numOfHiddenLayers+1] =
new LayerInfo(activationFunction, false, numOfOutput);
}
//store the information about the output 1 file type
if (d.ready()){
output1Type = Integer.valueOf(nextValidLine(d));
}
//store the information about the output 1 name
if (d.ready()){
output1Name = nextValidLine(d);
}
//store the information about the output 2 file type
if (d.ready()){
output2Type = Integer.valueOf(nextValidLine(d));
}
//store the information about the output 2 name
if (d.ready()){
output2Name = nextValidLine(d);
}
//Store the information for the desired training error
if (d.ready()){
desiredTrainingError = Double.valueOf(nextValidLine(d));
}
//Store the information for the maximum number of epochs
if (d.ready()){
numOfEpochs = Integer.valueOf(nextValidLine(d));
}
//Store the information for the desired network type
if (d.ready()){
networkType = Integer.valueOf(nextValidLine(d));
}
//We need additional variables if we are using Backpropagation
if (networkType == 1){
//Store the information for the learning rate
if (d.ready()){
learningRate = Double.valueOf(nextValidLine(d));
}
//Store the information for the momentum
if (d.ready()){
momentum = Double.valueOf(nextValidLine(d));
}
}
//TODO: reorder this
//output the information from the config file
System.out.println("config file validated:");
System.out.println("\tfilePath = " + filePath);
System.out.println("\thasHeaders = " + hasHeaders);
System.out.println("\tnumOfInput = " + numOfInput);
System.out.println("\tnumOfOutput = " + numOfOutput);
System.out.println("\tnumOfHiddenLayers = " + numOfHiddenLayers);
for (LayerInfo l: allMyLayers){
System.out.println("\t" + l.toString());
}
System.out.println("\tdesiredTrainingError = "
+ desiredTrainingError);
System.out.println("\tnumOfEpochs = " + numOfEpochs);
System.out.println("\tnetworkType = " + networkType);
if (networkType == 1){
System.out.println("\tlearningRate = " + learningRate);
System.out.println("\tmomentum = " + momentum);
}
System.out.println("\toutput2Type = " + output1Type);
System.out.println("\toutput2Name = " + output1Name);
System.out.println("\toutput2Type = " + output2Type);
System.out.println("\toutput2Name = " + output2Name);
}
catch (Exception e){
//TODO: create more detailed error messages, to see where the error
//occurred
System.out.println("Invalid config file");
e.printStackTrace();
}
System.out.println("");
}
}

