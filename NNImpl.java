/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	private double []outDelt = null;
	private double [] hidDelt = null;

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{
		//return -1;
		// TODO: add code here
		for(int k = 0; k < inputNodes.size()-1; k++){
			inputNodes.get(k).setInput(inst.attributes.get(k));
		}
		// Calculate outputs of inputs to each
		// hidden node (sum: wi*xi)
		for(int l = 0; l < hiddenNodes.size()-1; l++){
			hiddenNodes.get(l).calculateOutput();
		}
		// Calculate outputs of hidden nodes to
		// each output node (sum: wi*xi)
		for(int m = 0; m < outputNodes.size(); m++){
			outputNodes.get(m).calculateOutput();
		}
		double maxVal = 0.0;
		int maxValInd = 0;
		for(int i = 0; i < outputNodes.size(); i++){
			if(maxVal <= Math.round(outputNodes.get(i).getOutput()*10.0/10.0)){
				maxValInd = i;
				maxVal = outputNodes.get(i).getOutput();
			}
			//			System.out.print(" "+i+": "+outputNodes.get(i).getOutput()+" ");
		}//System.out.println();
		return maxValInd;
	}





	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		// repeat 'til max epoch
		for(int i = 0; i < maxEpoch; i++){
			// train network using all examples
			// (Propagate examples forward)
			for(int j = 0; j < trainingSet.size(); j++){
				// initialize backPropWeights
				hidDelt = new double[hiddenNodes.size()-1];
				outDelt = new double[outputNodes.size()];
				// set input values for particular example
				for(int k = 0; k < inputNodes.size()-1; k++){
					inputNodes.get(k).setInput(trainingSet.get(j).attributes.get(k));
				}
				// Calculate outputs of inputs to each
				// hidden node (sum: wi*xi)
				for(int l = 0; l < hiddenNodes.size()-1; l++){
					hiddenNodes.get(l).calculateOutput();
				}
				// Calculate outputs of hidden nodes to
				// each output node (sum: wi*xi)
				for(int m = 0; m < outputNodes.size(); m++){
					outputNodes.get(m).calculateOutput();
				}
				// get delta for output nodes
				for(int n = 0; n < outputNodes.size(); n++){
					double target = trainingSet.get(j).classValues.get(n);
					double actual = outputNodes.get(n).getOutput();
					double gPrime;// = 0.0;
					if(outputNodes.get(n).getSum() <= 0){
						gPrime = 0.0;
					}else{
						gPrime = 1.0;
					}
					// tanh gPrime
					//double gPrime = 1 - Math.pow(Math.tanh(outputNodes.get(n).getOutput()), 2);
					
					//sigmoid gPrime
					//double gPrime = outputNodes.get(n).getOutput()*(1-outputNodes.get(n).getOutput()); 
					
					outDelt[n] = (target - actual)*gPrime;
				}
				
				// get delta for hidden nodes
				for(int r = 0; r < hiddenNodes.size()-1; r++){
					double summation = 0.0;
					for(int p = 0; p < outputNodes.size(); p++){
						summation += outputNodes.get(p).parents.get(r).weight*outDelt[p];
					}
					double gPrime;// = 0.0;
					if(hiddenNodes.get(r).getSum() <= 0){
						gPrime = 0.0;
					}else{
						gPrime = 1.0;
					}
					// tanh gPrime
					//double gPrime = 1 - Math.pow(Math.tanh(hiddenNodes.get(r).getOutput()), 2);
					
					// sigmoid gPrime;
					//double gPrime = hiddenNodes.get(r).getOutput()*(1-hiddenNodes.get(r).getOutput());
					
					hidDelt [r] = summation*gPrime;
				}
				// Update weights btwn input and hidden
				for(int s = 0; s < hiddenNodes.size()-1; s++){
					for(int t = 0; t < hiddenNodes.get(s).parents.size()-1; t++){
						hiddenNodes.get(s).parents.get(t).weight += hiddenNodes.get(s).parents.get(t).node.getOutput()*learningRate*hidDelt[s];
					}
				}
				// Update weight btwn hidden and output
				for(int u = 0; u < outputNodes.size(); u++){
					for(int v = 0; v < outputNodes.get(u).parents.size()-1; v++){
						outputNodes.get(u).parents.get(v).weight += outputNodes.get(u).parents.get(v).node.getOutput()*learningRate*outDelt[u];
					}
				}
			}
//			//average gradient
//			// average weights btwn input and hidden
//			for(int s = 0; s < hiddenNodes.size()-1; s++){
//				for(int t = 0; t < hiddenNodes.get(s).parents.size()-1; t++){
//					hiddenNodes.get(s).parents.get(t).weight = hiddenNodes.get(s).parents.get(t).weight/trainingSet.size();
//				}
//			}
//			// average weights btwn hidden and output
//			for(int u = 0; u < outputNodes.size(); u++){
//				for(int v = 0; v < outputNodes.get(u).parents.size()-1; v++){
//					outputNodes.get(u).parents.get(v).weight = outputNodes.get(u).parents.get(v).weight/trainingSet.size();
//				}
//			}
//			//
//			for(int s = 0; s < hiddenNodes.size()-1; s++){
//				for(int t = 0; t < hiddenNodes.get(s).parents.size()-1; t++){
//					hiddenNodes.get(s).parents.get(t).weight += hiddenNodes.get(s).parents.get(t).node.getOutput()*learningRate*hidDelt[s];
//				}
//			}
//			// Update weight btwn hidden and output
//			for(int u = 0; u < outputNodes.size(); u++){
//				for(int v = 0; v < outputNodes.get(u).parents.size()-1; v++){
//					outputNodes.get(u).parents.get(v).weight += outputNodes.get(u).parents.get(v).node.getOutput()*learningRate*outDelt[u];
//				}
//			}
		}
	}
}
