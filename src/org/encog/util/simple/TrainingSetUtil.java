package org.encog.util.simple;

/*
* modified: added condition so that it will ignore rows with incomplete
data
 */

 /*
* Additional modification: added way to retrieve header information
 */
import java.util.ArrayList;
import java.util.List;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;

import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.util.EngineArray;
import org.encog.util.ObjectPair;
import org.encog.util.csv.CSVError;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class TrainingSetUtil {

	private static List<String> columnNames = new ArrayList<String>();

	/**
	 * Load a CSV file into a memory dataset.
	 *
	 * @param format The CSV format to use.
	 * @param filename The filename to load.
	 * @param headers True if there is a header line.
	 * @param inputSize The input size. Input always comes first in a file.
	 * @param idealSize The ideal size, 0 for unsupervised.
	 * @return A NeuralDataSet that holds the contents of the CSV file.
	 */
	public static MLDataSet loadCSVTOMemory(CSVFormat format,
		String filename, boolean headers, int inputSize, int idealSize) {
		MLDataSet result = new BasicMLDataSet();
		ReadCSV csv = new ReadCSV(filename, headers, format);

		if (headers) {
			columnNames = csv.getColumnNames();
		}

		int ignored = 0;

		while (csv.next()) {
			MLData input = null;
			MLData ideal = null;
			int index = 0;
			try {
				input = new BasicMLData(inputSize);
				for (int i = 0; i < inputSize; i++) {
					double d = csv.getDouble(index++);
					input.setData(i, d);
				}

				if (idealSize > 0) {

					ideal = new BasicMLData(idealSize);
					for (int i = 0; i < idealSize; i++) {
						double d = csv.getDouble(index++);
						ideal.setData(i, d);
					}
				}

				MLDataPair pair = new BasicMLDataPair(input, ideal);
				result.add(pair);
			} catch (CSVError e) {
				ignored++;

//e.printStackTrace();
			}
		}
		System.out.println("Rows ignored: " + ignored);

		return result;
	}

	public static ObjectPair<double[][], double[][]> trainingToArray(
		MLDataSet training) {
		int length = (int) training.getRecordCount();
		double[][] a = new double[length][training.getInputSize()];
		double[][] b = new double[length][training.getIdealSize()];

		int index = 0;
		for (MLDataPair pair : training) {
			EngineArray.arrayCopy(pair.getInputArray(), a[index]);
			EngineArray.arrayCopy(pair.getIdealArray(), b[index]);
			index++;
		}

		return new ObjectPair<double[][], double[][]>(a, b);
	}

	/**
	 * @return the columnNames
	 */
	public static List<String> getColumnNames() {
		return columnNames;
	}

}
