package timeseriesweka.classifiers;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.lazy.kNN;

import weka.core.*;

import weka.core.EuclideanDistance;
import timeseriesweka.elastic_distance_measures.DTW;
import utilities.ClassifierTools;

/* This class is a specialisation of kNN that can only be used with the efficient DTW distance
 * 
 * The reason for specialising is this class has the option of searching for the optimal window length
 * through a grid search of values.
 * 
 * By default this class does a search. 
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible window size, then sets the 
 * proportion to the one with the largest accuracy. This will be slow. Speed it up by
 * 
 * 1. Set the max window size to consider by calling
 * setMaxWindowSize(double r) where r is on range 0..1, with 1 being a full warp.
 * 
 * 2. Set the increment size 
 * setIncrementSize(int s) where s is on range 1...trainSetSize 
 * 
 * This is a basic brute force implementation, not optimised! There are probably ways of 
 * incrementally doing this. It could further be speeded up by using PAA to reduce the dimensionality first.
 * 
 */

public class DTW_kNN extends kNN {
	private boolean optimiseWindow=false;
	private double windowSize=0.1;
	private double maxWindowSize=1;
	private int incrementSize=10;
	private Instances train;
	private int trainSize;
	private int bestWarp;
	DTW dtw=new DTW();
	
//	DTW_DistanceEfficient dtw=new DTW_DistanceEfficient();
	public DTW_kNN(){
		super();
		dtw.setR(windowSize);
		setDistanceFunction(dtw);
		super.setKNN(1);
	}
	
	public void optimiseWindow(boolean b){ optimiseWindow=b;}
	public void setMaxR(double r){ maxWindowSize=r;}
	
	
	public DTW_kNN(int k){
		super(k);
		dtw.setR(windowSize);
		optimiseWindow=true;
		setDistanceFunction(dtw);
	}
	public void buildClassifier(Instances d){
		dist.setInstances(d);
		train=d;
		trainSize=d.numInstances();
		
		//System.out.print(trainSize);
		if(optimiseWindow){
			

			double maxR=0;
			double maxAcc=0;
/*Set the maximum warping window: Not this is all a bit mixed up. 
The window size in the r value is range 0..1, but the increments should be set by the 
data*/
			int dataLength=train.numAttributes()-1;
			int max=(int)(dataLength*maxWindowSize);
//			System.out.println(" MAX ="+max+" increment size ="+incrementSize);
			for(double i=0;i<max;i+=incrementSize){
				//Set r for current value
				dtw.setR(i/(double)dataLength);
				double acc=crossValidateAccuracy();
//				System.out.println("\ti="+i+" r="+(i/(double)dataLength)+" Acc = "+acc);
				if(acc>maxAcc){
					maxR=i/dataLength;
					maxAcc=acc;
//					System.out.println(" Best so far ="+maxR +" Warps ="+i+" has Accuracy ="+maxAcc);
				}
			}
			bestWarp=(int)(maxR*dataLength);
			dtw.setR(maxR);

//			System.out.println(" Best R = "+maxR+" Best Warp ="+bestWarp+" Size = "+(maxR*dataLength));
		}
/*		try {
			FileWriter fileWriter = new FileWriter("/home/hanwang/proj/sideatt/cloudsec/course624/distance/attackl3ddis4.txt");
			for(int i=0; i<100;i++) {
				// Then just use the normal kNN with the DTW distance. Not doing this any more because its slow!
						for(int j=100; j<200; j++) {
							
							fileWriter.write("pointA,"+i+",pointB,"+j+"distance,"+dtw.distance(d.instance(i), d.instance(j))+",label,"+d.instance(i).classValue()+",label,"+d.instance(j).classValue()+'\n');
						}//System.out.print("distance:"+dtw.distance(d.instance(2), d.instance(i+2))+"label: "+d.instance(2).classValue()+"label: "+d.instance(i+1).classValue()+'\n');
						}
			fileWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/

		super.buildClassifier(d);
	}
/* No need to do this, since we can use the IBk version, which should be optimised!
	public double classifyInstance(Instance d){
//Basic distance, with early abandon, which has not been implemented in the distance comparison.		This is only for nearest neighbour
		double minSoFar=Double.MAX_VALUE;
		double dist; int index=0;
		for(int i=0;i<train.numInstances();i++){
			dist=dtw.distance(train.instance(i),d,minSoFar);
			if(dist<minSoFar){
				minSoFar=dist;
				index=i;
			}
		}
		return train.instance(index).classValue();
	}
*/
//Could do this for BER instead	
	private double crossValidateAccuracy(){
		double a=0,d=0, minDist;
		int nearest=0;
		Instance inst;
		for(int i=0;i<trainSize;i++){
//Find nearest to element i
			nearest=0;
			minDist=Double.MAX_VALUE;
			inst=train.instance(i);
			for(int j=0;j<trainSize;j++){
				if(i!=j){
//					if(i==0&&j<2)
//						System.out.println("\t"+inst+" and \n\t"+train.instance(j)+"\n\t\t ="+d);
					d=dtw.distance(inst,train.instance(j),minDist);
					if(d<minDist){
						nearest=j;
						minDist=d;
					}
				}
			}
//			System.out.println("\t\tDistance between "+i+" and "+nearest+" ="+minDist);
			
			//Measure accuracy for nearest to element i			
			if(inst.classValue()==train.instance(nearest).classValue())
				a++;
		}
		return a/(double)trainSize;
	}
	
	
	public static void main(String[] args) {
		for(int i=1;i<2;i++) {
		DTW_kNN c = new DTW_kNN();
		String path="/home/hanwang/Downloads/knn.txt";
		//Instances train = loadData("/home/hwang31/dev/instancemodeldata5/instancemodeldata5_TRAIN.arff");
		Instances train=loadData("/home/hanwang/Downloads/60samplestrain-5.arff");
		System.out.println("Attribute:" +i);
		Instances test10=loadData("/home/hanwang/Downloads/60samplestest.arff");
		/*Instances test50=loadData("/home/hwang31/proj/sideatt/cloudsec/attdata/matchrun/instancebased/win-50/bat"+i+".arff");
		Instances test100=loadData("/home/hwang31/proj/sideatt/cloudsec/attdata/matchrun/instancebased/win-100/bat"+i+".arff");
		Instances test200=loadData("/home/hwang31/proj/sideatt/cloudsec/attdata/matchrun/instancebased/win-200/bat"+i+".arff");*/
		train.setClassIndex(train.numAttributes()-1);
		System.out.println(train.numAttributes()-1);
		c.buildClassifier(train);
		try {
			ClassifierTools.accuracy(test10, c, "knn.txt");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//System.out.println("test2");
		try {
			System.out.println(ClassifierTools.accuracy(test10, c,"knn.txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println(c.crossValidateAccuracy());
		/*try {
			long start = System.nanoTime();
			System.out.println(ClassifierTools.accuracy(test10, c,"dtw200-"+i+".txt"));
			double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            //System.out.println("Testing done (" + testTime + "s)");
			//System.out.println(ClassifierTools.accuracy(test100, c,"dtw100-"+i+".txt"));
			//System.out.println(ClassifierTools.accuracy(test200, c,"dtw200-"+i+".txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/}
	}
	public static Instances loadData(String fileName)
	{
		Instances data=null;
		try{
			FileReader r;
			r= new FileReader(fileName); 
			data = new Instances(r); 

			data.setClassIndex(data.numAttributes()-1);
		}catch(Exception e)
		{
			System.out.println(" Error ="+e+" in method loadData");
		}
		return data;
	}

}
