package skynet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import org.encog.engine.network.activation.*;

/**
 * A child class out OutputWriter, used for creating .java files.
 *
 * @author bwinrich
 */
public class OutputWriterJava extends OutputWriter {

	private boolean standalone;

	/**
	 * Default constructor
	 */
	public OutputWriterJava() {
	}

	/**
	 * Constructor with parameter
	 *
	 * @param standalone main method or not
	 */
	public OutputWriterJava(boolean standalone) {
		this.standalone = standalone;
	}

	/* (non-Javadoc)
* @see OutputWriter#writeFile()
	 */
	@Override
	public void writeFile() {

		writeTwo("import java.lang.Math;");

		writeTwo("");

		writeTwo("public class " + outputName);
		writeTwo("{");

		writeTwo("\t//Variable declarations");

//Variables from headers of csv file, if applicable
		if (hasHeaders) {
			writeTwo("\t//Header Names");
			for (String s : columnNames) {
				writeTwo("\tpublic static double " + s + ";");
			}
		}

//variables - input layer
		writeTwo("\t//Input layer");
		for (int i = 0; i < inputCount; i++) {
			writeTwo("\tpublic static double i" + i + ";");
		}
		for (int i = inputCount; i < numberOfTotalNeurons[0]; i++) {
			writeTwo("\tprivate static double i" + i + " = "
				+ biases[biases.length - 1] + ";");
		}

//variables - hidden layers
		writeTwo("\t//Hidden layer(s)");
		for (int i = 1; i < layers - 1; i++) {
			writeTwo("\t//Hidden Layer " + i);
			for (int j = 0; j < numberOfNormalNeurons[i]; j++) {
				for (int k = 0; k < numberOfTotalNeurons[i - 1]; k++) {
					writeTwo("\tprivate static double h" + i + "n" + j + "f" + k
						+ ";");
				}
				writeTwo("\tprivate static double h" + i + "n" + j + "t;");
				writeTwo("\tprivate static double h" + i + "n" + j + ";");
			}
			for (int j = numberOfNormalNeurons[i]; j < numberOfTotalNeurons[i];
				j++) {
				writeTwo("\tprivate static double h" + i + "n" + j + " = "
					+ biases[biases.length - i - 1] + ";");
			}
		}

//varibles - output layer
		writeTwo("\t//Output layer");

		for (int i = 0; i < outputCount; i++) {
			for (int j = 0; j < numberOfTotalNeurons[layers - 2]; j++) {
				writeTwo("\tprivate static double o" + i + "f" + j + ";");
			}
			writeTwo("\tprivate static double o" + i + "t;");
			writeTwo("\tpublic static double o" + i + ";");
		}

		writeTwo("");

//standalone files will have a main method
		if (standalone) {

//TODO: customize this (from Tiberius)
			writeTwo("\tpublic static void main(String[] args)");
			writeTwo("\t{");
			writeTwo("\t\tinitData();");
			writeTwo("\t\tcalcNet();");
			for (int i = 0; i < outputCount; i++) {
				writeTwo("\t\tSystem.out.println(o" + i + ");");
			}
			writeTwo("\t}");
			writeTwo("");

//TODO: customize this (from Tiberius)
			writeTwo("\tpublic static void initData()");
			writeTwo("\t{");
			writeTwo("\t\t//data is set here");
			for (int i = 0; i < inputCount; i++) {
				if (hasHeaders) {
					writeTwo("\t\t" + columnNames[i] + " = 1;");
				} else {
					writeTwo("\t\ti" + i + " = 1;");
				}
			}
			writeTwo("\t}");
			writeTwo("");
		}

		double weight;
		String sum = "";

		writeTwo("\tpublic static void calcNet()");
		writeTwo("\t{");

//Some extra code if we have headers, to set the default input
//variables to the header variables
		if (hasHeaders) {
			for (int i = 0; i < inputCount; i++) {
				writeTwo("\t\ti" + i + " = " + columnNames[i] + ";");
			}
		}

//Hidden layers calculation
		for (int i = 1; i < layers - 1; i++) {
			for (int j = 0; j < numberOfNormalNeurons[i]; j++) {
				writeTwo("");

				sum = "";

				for (int k = 0; k < numberOfTotalNeurons[i - 1]; k++) {
					weight = network.getWeight(i - 1, k, j);
					if (i == 1) {
						writeTwo("\t\th" + i + "n" + j + "f" + k + " = i" + k
							+ " * " + weight + ";");
					} else {
						writeTwo("\t\th" + i + "n" + j + "f" + k + " = h" + (i - 1)
							+ "n" + k + " * " + weight + ";");
					}

					if (k == 0) {
						sum = "h" + i + "n" + j + "f" + k;
					} else {
						sum += " + h" + i + "n" + j + "f" + k;
					}
				}

				writeTwo("\t\th" + i + "n" + j + "t = " + sum + ";");

				String af = parseActivationFunction(network.getActivation(i),
					"h" + i + "n" + j + "t", "h" + i + "n" + j);
				writeTwo("\t\t" + af);
			}
		}

//Output layer calculation
		writeTwo("");

		sum = "";

		for (int i = 0; i < outputCount; i++) {

			for (int j = 0; j < numberOfTotalNeurons[layers - 2]; j++) {
				weight = network.getWeight(layers - 2, j, i);
				writeTwo("\t\to" + i + "f" + j + " = h" + (layers - 2) + "n" + j
					+ " * " + weight + ";");

				if (j == 0) {
					sum = "o" + i + "f" + j;
				} else {
					sum += " + o" + i + "f" + j;
				}
			}
			writeTwo("\t\to" + i + "t = " + sum + ";");

			String af = parseActivationFunction(network.getActivation(layers - 1),
				"o" + i + "t", "o" + i);
			writeTwo("\t\t" + af);
		}

//Some extra code if we have headers, to set the default input
//variables to the header variables
		if (hasHeaders) {
			writeTwo("");
			for (int i = 0; i < outputCount; i++) {
				writeTwo("\t\t" + columnNames[i + inputCount] + " = o" + i
					+ ";");
			}
		}

		writeTwo("\t}");

		writeTwo("");
		writeTwo("}");

		try {
			if (bw2 != null) {
				bw2.close();
			}
		} catch (Exception ex) {
			System.out.println("Error in closing the BufferedWriter" + ex);
		}
	}

	/* (non-Javadoc)
* @see OutputWriter#createFile(java.lang.String)
	 */
	@Override
	public void createFile(String output2Name) {

		outputName = output2Name;

		try {
			file2 = new File(output2Name + ".java");
			if (!file2.exists()) {
				file2.createNewFile();
			}

			FileWriter fw2 = new FileWriter(file2);
			bw2 = new BufferedWriter(fw2);
		} catch (IOException e) {
// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	protected String parseActivationFunction(ActivationFunction af,
		String varName, String targetVarName) {
		String text = null;

		if (af instanceof ActivationSigmoid) {
			text = targetVarName + " = 1.0 / (1.0 + Math.exp(-1 * " + varName
				+ "));";
		} else if (af instanceof ActivationTANH) {
			text = targetVarName + " = Math.tanh(" + varName + ");";
		} else if (af instanceof ActivationLinear) {
			text = targetVarName + " = " + varName + ";";
		} else if (af instanceof ActivationElliott) {
			double s = af.getParams()[0];
			text = targetVarName + " = ((" + varName + " * " + s
				+ ") / 2) / (1 + Math.abs(" + varName + " * " + s
				+ ")) + 0.5;";
		} else if (af instanceof ActivationGaussian) {
			text = targetVarName + " = Math.exp(-Math.pow(2.5*" + varName
				+ ",2.0));";
		} else if (af instanceof ActivationLOG) {
			text = "if(" + varName + " >= 0){\n\t" + targetVarName
				+ " = Math.log(1 + " + varName + ");\n}else{\n\t"
				+ targetVarName + " = -Math.log(1 - " + varName + ");\n}";
		} else if (af instanceof ActivationRamp) {
			double paramRampHighThreshold
				= ((ActivationRamp) (af)).getThresholdHigh();
			double paramRampLowThreshold
				= ((ActivationRamp) (af)).getThresholdLow();

			double paramRampHigh = ((ActivationRamp) (af)).getHigh();
			double paramRampLow = ((ActivationRamp) (af)).getLow();
			double slope = (paramRampHighThreshold - paramRampLowThreshold)
				/ (paramRampHigh - paramRampLow);

			text = "if(" + varName + " < " + paramRampLowThreshold + ") {\n\t"
				+ targetVarName + " = " + paramRampLow + ";\n} else if ("
				+ varName + " > " + paramRampHighThreshold + ") {\n\t"
				+ targetVarName + " = " + paramRampHigh + ";\n} else {\n\t"
				+ targetVarName + " = (" + slope + " * " + varName + ");";
		} else if (af instanceof ActivationSIN) {
			text = targetVarName + " = Math.sin(2.0*" + varName + ");";
		} else if (af instanceof ActivationStep) {
			double paramStepCenter = ((ActivationStep) (af)).getCenter();
			double paramStepLow = ((ActivationStep) (af)).getLow();
			double paramStepHigh = ((ActivationStep) (af)).getHigh();

			text = "if (" + varName + ">= " + paramStepCenter + ") {\n\t"
				+ targetVarName + " = " + paramStepHigh + ";\n} else {\n\t"
				+ targetVarName + " = " + paramStepLow + ";\n}";
		} else if (af instanceof ActivationBiPolar) {
			text = "if(" + varName + " > 0) {\n\t" + targetVarName
				+ " = 1;\n} else {\n\t" + targetVarName + " = -1;\n}";
		} else if (af instanceof ActivationBipolarSteepenedSigmoid) {
			text = targetVarName + " = (2.0 / (1.0 + Math.exp(-4.9 * "
				+ varName + "))) - 1.0;";
		} else if (af instanceof ActivationClippedLinear) {

			text = "if(" + varName + " < -1.0) {\n\t" + targetVarName
				+ " = -1.0;\n} else if (" + varName + " > 1.0) {\n\t"
				+ targetVarName + " = 1.0;\n} else {\n\t" + targetVarName
				+ " = " + varName + ";\n}";
		} else if (af instanceof ActivationElliottSymmetric) {
			double s = af.getParams()[0];
			text = targetVarName + " = (" + varName + "*" + s
				+ ") / (1 + Math.abs(" + varName + "*" + s + "));";
		} else if (af instanceof ActivationSteepenedSigmoid) {
			text = targetVarName + " = 1.0 / (1.0 + Math.exp(-4.9 * " + varName
				+ "));";
		} else {
//Unimplemented activation function: Softmax (complicated)
//Unimplemented activation function: Competitive (complicated,
//non-differentiable)
//in Encog 3.3 there aren’t any other activation functions, so
//unless someone implements their own we shouldn’t get to this point
			text = "Error: unknown activation function";

		}
		return text;
	}
}
