import java.io.File;
import java.io.IOException;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;


public class Main {

	public static void main(String[] args) {
		
		try {
			Dataset data = FileHandler.loadDataset(new File("desp.txt"), 9, ",");
			Classifier knn = new KNearestNeighbors(5);
			knn.buildClassifier(data);
			Dataset dataForClassification = FileHandler.loadDataset(new File("Test.txt"), 9, ",");
			 
			Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
			for(Object o:pm.keySet())
			    System.out.println(o+": "+pm.get(o).getAccuracy());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
