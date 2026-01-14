import java.util.*;
import java.io.*;
import java.lang.Math.*; // static import가 아니면 .* 로 충분합니다.
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

// 파일 이름이 Main.java라면 클래스 이름도 Main이어야 합니다.
public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("DL4J MNIST 다운로드 시작...");

        int batchSize = 64; // 한 번에 가져올 사진 수
        
        // true = 학습용 데이터(60,000장), false = 테스트용 데이터(10,000장)
        // seed는 랜덤 시드 (123)
        // ★ 이 줄이 실행될 때 인터넷에서 파일을 다운로드합니다 (처음 1회만)
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 123);
        
        System.out.println("다운로드 및 로드 완료!");
        System.out.println("Hello, DL4J World!");
    }
}