import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * @author hizukuri
 *
 */
public class Homo {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		String bookObject = "/home/hizukuri/Pictures/CIMG0056.JPG";
		String bookScene = "/home/hizukuri/Pictures/CIMG0057.JPG";


		System.out.println("Started....");
		System.out.println("Loading images...");
		//Mat readi = Highgui.imread(bookObject, Highgui.CV_LOAD_IMAGE_COLOR);
		//Mat objectImage = new Mat(readi, new Rect(new double[] {100,200,200,200}));
		Mat objectImage = Highgui.imread(bookObject, Highgui.CV_LOAD_IMAGE_COLOR);
		Mat sceneImage = Highgui.imread(bookScene, Highgui.CV_LOAD_IMAGE_COLOR);
		
		if(objectImage.empty() || sceneImage.empty()){
			System.out.println("File Not found!");
		}

		Mat result = createPartSub(objectImage, sceneImage, 500, 500, 500, 500);
		Highgui.imwrite("./imgs/output/sabun.jpg", result);
		
		Imgproc.threshold(result, result, /*127*/0.0,255.0,Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
		//ノイズ除去
		Imgproc.erode(result, result, new Mat(), new Point(-1,-1), 1);
		//欠損部補完
		Imgproc.dilate(result, result, new Mat());
		Highgui.imwrite("./imgs/output/zentai.jpg", result);
		
		Mat a = new Mat();
		match(sceneImage, objectImage, a);
		Highgui.imwrite("./imgs/output/zentai2.jpg", a);
	}


	/**
	 * @param past 過去の写真(ホモグラフィ変形後)
	 * @param src2　現在の写真
	 * @param width 部分差分画像の横幅
	 * @param height　部分差分画像の縦幅
	 * @param x 差分画像のx移動 今のところx,yのパラメータはwidthとheightと同じにする(重なり部分の処理未実装のため)
	 * @param y 差分画像のy移動 
	 * @return
	 */
	public static Mat createPartSub(Mat past, Mat present, int width, int height, int x, int y){
		//結果出力用mat
		Mat baseimg = imageDiff(past, present);
	
		//作業用mat
		Mat target = new Mat();


		int i = 0;
		int j = 50;
		int w=width;
		boolean flagx = true;
		boolean flagy = true;
		Rect box;

		while(i<present.cols()||flagx){
			if(i+width>present.cols()){
				w=present.cols()-i;
				flagx=false;
			}
			j=0;
			int h = height;
			flagy=true;
			while(j<present.rows()&&flagy){

				if(j+height>present.rows()){
					h=present.rows()-j;
					flagy=false;

				}
				System.out.println(i+" "+ j +" "+w+" "+h);
				box = new Rect(new double[] {i,j,w,h});
				//ターゲット画像の切り出し
				target = new Mat(present,box);

				//対応画像の切り出し
				Mat tmp = new Mat();
				//マッチングの結果があるなら
				if(match(target,past,tmp)){
					//ターゲット画像と対応画像の差分
					Mat diff = imageDiff(target, tmp);
					//ベース画像の位置
					Mat Roi= new Mat(baseimg, box);
					//画像の貼り付け
					diff.copyTo(Roi);
					//画像の左上角のy座標増
				}
				j= j+y;
			}
			//画像の左上角のx座標増
			i=i+x;
		}

		return baseimg;
	}

	/**
	 * @param objectImage オブジェクトのmat
	 * @param sceneImage  シーンのmat
	 * @return
	 */
	public static boolean match(Mat objectImage, Mat sceneImage, Mat dst){


		System.out.println("Started....");
		System.out.println("Loading images...");
		//SURF特徴
		MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		System.out.println("Detecting key points...");
		featureDetector.detect(objectImage, objectKeyPoints);
		KeyPoint[] keypoints = objectKeyPoints.toArray();
		System.out.println(keypoints);
		//特徴記述
		MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		System.out.println("Computing descriptors...");
		descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

		// Create the matrix for output image.
		Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
		Scalar newKeypointColor = new Scalar(255, 0, 0);

		System.out.println("Drawing key points on object image...");
		Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

		// Match object image with the scene image
		MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
		MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
		System.out.println("Detecting key points in background image...");
		featureDetector.detect(sceneImage, sceneKeyPoints);
		System.out.println("Computing descriptors in background image...");
		descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);

		Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
		Scalar matchestColor = new Scalar(0, 255, 0);

		List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		System.out.println("Matching object and scene images...");
		descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);

		//System.out.println("Calculating good match list...");
		LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

		float nndrRatio = 0.7f;

		for (int i = 0; i < matches.size(); i++) {
			MatOfDMatch matofDMatch = matches.get(i);
			DMatch[] dmatcharray = matofDMatch.toArray();
			DMatch m1 = dmatcharray[0];
			DMatch m2 = dmatcharray[1];

			if (m1.distance <= m2.distance * nndrRatio) {
				goodMatchesList.addLast(m1);

			}
		}

		System.out.println(goodMatchesList.size());
		if (goodMatchesList.size() >= 20) {
			System.out.println("Object Found!!!");

			List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
			List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

			LinkedList<Point> objectPoints = new LinkedList<>();
			LinkedList<Point> scenePoints = new LinkedList<>();

			for (int i = 0; i < goodMatchesList.size(); i++) {
				objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
				scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
			}

			MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
			objMatOfPoint2f.fromList(objectPoints);
			MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
			scnMatOfPoint2f.fromList(scenePoints);

			Mat homography = Calib3d.findHomography(scnMatOfPoint2f,objMatOfPoint2f, Calib3d.RANSAC, 3);

			System.out.println("Drawing matches image...");
			MatOfDMatch goodMatches = new MatOfDMatch();
			goodMatches.fromList(goodMatchesList);

			Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);

			Size s = new Size(objectImage.cols(),objectImage.rows());

			Imgproc.warpPerspective(sceneImage, dst, homography, s);

		} else {
			System.out.println("Object Not Found");
			return false;
		}

		System.out.println("Ended....");
		return true;
	}

	public static Mat imageDiff(Mat src1, Mat src2){
		Mat result = new Mat();
		Mat gray_img1 = new Mat();
		Mat gray_img2 = new Mat();
		//グレースケールに変換
		Imgproc.cvtColor(src1, gray_img1, Imgproc.COLOR_RGBA2GRAY);
		Imgproc.cvtColor(src2, gray_img2, Imgproc.COLOR_RGBA2GRAY);

		//差分
		Core.absdiff(gray_img1, gray_img2, result);
		return result;
	}

}


