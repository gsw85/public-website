package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * Created by susaneraly on 6/9/16.
 */
public class ImagePipelineExample {

    protected static final Logger log = LoggerFactory.getLogger(ImagePipelineExample.class);

    //Images are of format given by allowedExtension -
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    protected static final long seed = 12345;

    public static final Random randNumGen = new Random(seed);

    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int outputNum = 2;

    public static void main(String[] args) throws Exception {

        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory
        //
        //
        File parentDir = new File("/home/agibsonccc/Documents/isoftstone-data");
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadocs for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        //Often there is a need to transforming images to artificially increase the size of the dataset
        //DataVec has built in powerful features from OpenCV
        //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10));
            */

        //You can use the ShowImageTransform to view your images
        //Code below gives you a look before and after, for a side by side comparison
        ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));

        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData,transform);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);

        int nChannels = 3;
        int iterations = 1;

        int rows = 28;
        int cols = 28;

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.0005)
            .learningRate(0.01)//.biasLearningRate(0.02)
            //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20)
                .activation("identity")
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                //Note that nIn needed be specified in later layers
                .stride(1, 1)
                .nOut(50)
                .activation("identity")
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(4, new DenseLayer.Builder().activation("relu")
                .nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation("softmax")
                .build())
            .backprop(true).pretrain(false);
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder,rows,cols,nChannels);

        MultiLayerNetwork network = new MultiLayerNetwork(builder.build());
        network.init();
        network.fit(dataIter);
    }
}
