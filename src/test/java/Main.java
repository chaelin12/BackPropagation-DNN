import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("DL4J MNIST 다운로드 시작...");

        // 1. 데이터 준비
        int batchSize = 64; 
        int rngSeed = 123;
        int numEpochs = 5; // 전체 데이터를 5번 반복 학습

        System.out.println("데이터 로드 중...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        //2.네트워크 구성
        MultiLayerConfiguration conf
          = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)//랜덤 시드
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))//학습률
                .l2(0.0001)//정규화
                .list()
                //은닉층
                .layer(0, new DenseLayer.Builder()
                    .nIn(784)
                    .nOut(256)
                    .activation(Activation.SIGMOID)
                    .build())
                //출력층
                .layer(1, new OutputLayer.Builder(
                    LossFunctions.LossFunction.MCXENT)//목적함수 계산
                    .nIn(256)
                    .nOut(10)
                    .activation(Activation.SOFTMAX)
                    .build())
                .build();  

        //3.네트워크 생성 및 훈련
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //UI 도구 사용
        System.out.println("UI 서버 시작 중...");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        System.out.println("훈련 시작");
        for(int i=0;i<numEpochs;i++){
            model.fit(mnistTrain);
            mnistTrain.reset();
            System.out.println((i+1)+"epoch 1 완료;");
        }

        //4. test
        Evaluation eval = new Evaluation(10);
        while(mnistTest.hasNext()){
            DataSet t = mnistTest.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);

        }
        System.out.println(eval.stats());
        
        
    }
}