import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Comparator;
import java.util.Collections;
import java.io.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import javax.swing.*;
import javafx.util.Pair;
import org.opencv.core.*;
import org.opencv.videoio.Videoio;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.calib3d.Calib3d;

class HexagonObject {

	private MatOfPoint3f objectPoints;

	public HexagonObject() {
		List<org.opencv.core.Point3> pts = new ArrayList<org.opencv.core.Point3>();
		double theta=Math.PI*4/3;
		for(int i=0; i<6; ++i) {
			pts.add(new org.opencv.core.Point3(17.625 * Math.cos(theta), 17.625 * Math.sin(theta), 0));
			theta+=Math.PI/3;
		}
		/*pts.add(new org.opencv.core.Point3(-1,-2.2,0));
		pts.add(new org.opencv.core.Point3(-2.1,0,0));
		pts.add(new org.opencv.core.Point3(-1,2.4,0));
		pts.add(new org.opencv.core.Point3(1.1,2.7,0));
		pts.add(new org.opencv.core.Point3(2.6,0.8,0));
		pts.add(new org.opencv.core.Point3(1.4,-1.8,0));*/
		
		objectPoints = new MatOfPoint3f();
	  objectPoints.fromList(pts);
	}

	public MatOfPoint3f getObjectPoints() {
		return this.objectPoints;
	}

	public MatOfPoint3f getObjectPoints(double zOffset) {
		List<org.opencv.core.Point3> pts = new ArrayList<org.opencv.core.Point3>();
		for(org.opencv.core.Point3 pt: this.objectPoints.toList()) {
			org.opencv.core.Point3 result = new org.opencv.core.Point3();
			result.x = pt.x;
			result.y = pt.y;
			result.z = pt.z + zOffset;
			pts.add(result);
		}
		MatOfPoint3f ret = new MatOfPoint3f();
		ret.fromList(pts);
		return ret;
	}
};

class Vector2 {
	public double x;
	public double y;

	public Vector2(double x, double y) {
		this.x = x;
		this.y = y;
	}

	public double dot(Vector2 other) {
		return this.x * other.x + this.y * other.y;
	}

	public double length() {
		return Math.sqrt(x * x + y * y);
	}
}

class VisionObject {
	public Mat rvec;
	public Mat tvec;
	public int index;

	public VisionObject(int index, Mat rvec, Mat tvec) {
		this.index = index;
		this.rvec = rvec;
		this.tvec = tvec;
	}
}

class PolygonTracker {

	public static final String TITLE = "Test";
	public static final int WIDTH = 840;
	public static final int HEIGHT = 680;

	public static int GAUSSIAN_BLUR_KERNEL_SIZE = 7;
	public static double GAUSSIAN_X_STDDEV = 0;

	private VideoCapture vc;
	private int nFrames;
	private List<MatOfPoint> currHexagons;
	private List<MatOfPoint> currOctagons;

	private JFrame jframe;
	public TestPanel panel;

	public PolygonTracker() {
		this("", 0);
	}

	public PolygonTracker(String filename, int index) {
		if(filename.length() == 0) {
			vc = new VideoCapture(index);
		}
		else {
			vc = new VideoCapture(filename);
		}

		jframe = new JFrame(PolygonTracker.TITLE);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setSize(PolygonTracker.WIDTH, PolygonTracker.HEIGHT);

		panel = new TestPanel();
		jframe.add(panel);
		jframe.setVisible(true);
	}

	// https://stackoverflow.com/questions/30258163/display-image-in-mat-with-jframe-opencv-3-00
	public static BufferedImage bufferedImage(Mat m) {
    int type = BufferedImage.TYPE_BYTE_GRAY;
    if (m.channels() > 1) {
			type = BufferedImage.TYPE_3BYTE_BGR;
    }
    BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
    m.get(0,0,((DataBufferByte)image.getRaster().getDataBuffer()).getData()); // get all the pixels
    return image;
	}

	public static double calculateDistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
		return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	}

	public static boolean hexagonTest(MatOfPoint hexagon) {
		List<org.opencv.core.Point> pts = hexagon.toList();
		int index = 0;
		for(int i=1;i<6;i++) {
			if(pts.get(i).x < pts.get(index).x) { index = i; }
		}
		for(int i=0;i<6;i++) {
			if(i == index) { continue; }
			if(pts.get(i).x < pts.get(index).x) { return false; }
			if(pts.get(i).x > pts.get((index + 3) % 6).x) { return false; }
		}
		for(int i=0;i<6;i++) {
			if(i == (index + 4) % 6 || i == (index + 5) % 6) { continue; }
			if(pts.get(i).y < pts.get((index + 4) % 6).y) { return false; }
			if(pts.get(i).y < pts.get((index + 5) % 6).y) { return false; }
		}
		for(int i=0;i<6;i++) {
			if(i == (index + 1) % 6 || i == (index + 2) % 6) { continue; }
			if(pts.get(i).y > pts.get((index + 1) % 6).y) { return false; }
			if(pts.get(i).y > pts.get((index + 2) % 6).y) { return false; }
		}
		/*double x2 = hexagon.toList().get(1).x;
		for(int i=0;i<6;i++) {
			if(i == 1) { continue; }
			if(x2 > hexagon.toList().get(i).x) { return false; }
		}
		double x5 = hexagon.toList().get(4).x;
		for(int i=0;i<6;i++) {
			if(i == 4) { continue; }
			if(x5 < hexagon.toList().get(i).x) { return false; }
		}
		double y1 = hexagon.toList().get(0).y;
		double y6 = hexagon.toList().get(5).y;
		for(int i=1;i<5;i++) {
			if(y1 > hexagon.toList().get(i).y) { return false; }
			if(y6 > hexagon.toList().get(i).y) { return false; }
		}
		double y3 = hexagon.toList().get(2).y;
		double y4 = hexagon.toList().get(3).y;
		for(int i=0;i<6;i++) {
			if(i == 2 || i == 3) { continue; }
			if(y3 < hexagon.toList().get(i).y) { return false; }
			if(y4 < hexagon.toList().get(i).y) { return false; }
		}*/
		return true;
	}

	public static Pair<Double, MatOfPoint> processHexagon(MatOfPoint polygon) {
		// Sort vertices by angle. Bubble up points with least colinear angles to top.
    // And only retain top 6 vertices.
		org.opencv.core.Point p1, p2, p3;
		Vector2 v12, v23;
		ArrayList<Pair<Double, Integer>> pairs = new ArrayList<Pair<Double, Integer>>();
		int numPassed = 0;
		for(int i=0;i<polygon.toList().size();i++) {
			p1 = polygon.toList().get(i);
			p2 = polygon.toList().get((i+1)%polygon.toList().size());
			p3 = polygon.toList().get((i+2)%polygon.toList().size());

			v12 = new Vector2(p2.x - p1.x, p2.y - p1.y);
			v23 = new Vector2(p3.x - p2.x, p3.y - p2.y);

			double dotProduct = v12.dot(v23);
			double lengths = v12.length() * v23.length();

			if(lengths == 0) { continue; }
			double cosTheta = dotProduct / lengths;

			pairs.add(new Pair<Double, Integer>(cosTheta, (i + 1) % polygon.toList().size()));
		}

		// Sort based on cosine of angle.
		// https://stackoverflow.com/questions/29920027/how-can-i-sort-a-list-of-pairstring-integer
		Collections.sort(pairs, new Comparator<Pair<Double, Integer>>() {
			@Override public int compare(final Pair<Double, Integer> p1, final Pair<Double, Integer> p2) {
				return Double.compare(p1.getKey().doubleValue(), p2.getKey().doubleValue());
			}
		});

		// Add indices of top six.
		ArrayList<Integer> result = new ArrayList<Integer>();
		for(int i=0;i<6;i++) {
			result.add(pairs.get(i).getValue().intValue());
		}
		// Sort indices so we add vertices in original order.
		Collections.sort(result);

		// Create new contour with only the top six points.
		ArrayList<org.opencv.core.Point> result2 = new ArrayList<org.opencv.core.Point>();
		for(int i=0;i<6;i++) {
			result2.add(polygon.toList().get(result.get(i)));
		}

		// Sort all contours by max-of-shortest line length.
		MatOfPoint result3 = new MatOfPoint();
		result3.fromList(result2);

		ArrayList<Double> distances = new ArrayList<Double>();
		for(int i=0;i<result2.size();i++) {
			p1 = result2.get(i);
			p2 = result2.get((i+1)%result2.size());
			distances.add(calculateDistance(p1, p2));
		}

		Collections.sort(distances);
		
		return new Pair<Double, MatOfPoint>(distances.get(0), result3);
	}

	public static Pair<Double, MatOfPoint> processOctagon(MatOfPoint polygon) {
		org.opencv.core.Point p1, p2, p3;
		Vector2 v12, v23;
		ArrayList<Pair<Double, Integer>> pairs = new ArrayList<Pair<Double, Integer>>();
		int numPassed = 0;
		for(int i=0;i<polygon.toList().size();i++) {
			p1 = polygon.toList().get(i);
			p2 = polygon.toList().get((i+1)%polygon.toList().size());
			p3 = polygon.toList().get((i+2)%polygon.toList().size());

			v12 = new Vector2(p2.x - p1.x, p2.y - p1.y);
			v23 = new Vector2(p3.x - p2.x, p3.y - p2.y);

			double dotProduct = v12.dot(v23);
			double lengths = v12.length() * v23.length();

			if(lengths == 0) { continue; }
			double cosTheta = dotProduct / lengths;

			pairs.add(new Pair<Double, Integer>(cosTheta, (i + 1) % polygon.toList().size()));
		}

		// https://stackoverflow.com/questions/29920027/how-can-i-sort-a-list-of-pairstring-integer
		Collections.sort(pairs, new Comparator<Pair<Double, Integer>>() {
			@Override public int compare(final Pair<Double, Integer> p1, final Pair<Double, Integer> p2) {
				return Double.compare(p1.getKey().doubleValue(), p2.getKey().doubleValue());
			}
		});

		if(polygon.toList().size() < 8) {
			return null;
		}

		ArrayList<Integer> result = new ArrayList<Integer>();
		for(int i=0;i<8;i++) {
			result.add(pairs.get(i).getValue().intValue());
		}

		Collections.sort(result);

		ArrayList<org.opencv.core.Point> result2 = new ArrayList<org.opencv.core.Point>();
		for(int i=0;i<8;i++) {
			result2.add(polygon.toList().get(result.get(i)));
		}

		MatOfPoint result3 = new MatOfPoint();
		result3.fromList(result2);

		ArrayList<Double> distances = new ArrayList<Double>();
		for(int i=0;i<result2.size();i++) {
			p1 = result2.get(i);
			p2 = result2.get((i+1)%result2.size());
			distances.add(calculateDistance(p1, p2));
		}

		Collections.sort(distances);
		
		return new Pair<Double, MatOfPoint>(distances.get(0), result3);
	}

	public static int polygonTests(MatOfPoint polygon, boolean noisy) {
		double totalDistance = calculateDistance(polygon.toList().get(0), polygon.toList().get(polygon.toList().size()-1)); 
		ArrayList<Double> distances = new ArrayList<Double>();
		for(int i=0;i<polygon.toList().size()-1;i++) {
			double d = calculateDistance(polygon.toList().get(i), polygon.toList().get(i+1));
			distances.add(d);
			totalDistance += d;
		}
		double averageDistance = totalDistance / polygon.toList().size();
		boolean flag = true;
		for(int i=0;i<distances.size();i++) {
			if(Math.abs(averageDistance - distances.get(i)) > averageDistance * 4 / 3) {
				flag = false;
				break;
			}
		}
		
		/*if(!flag) {
			if(noisy) {
				System.out.println("Failed side length test.");
			}
			return false;
		}*/
		if(!Imgproc.isContourConvex(polygon)) {
			if(noisy) {
				System.out.println("Failed convexity test.");
			}
			return -1;
		}
		/*if(!anglesTest(polygon)) {
			return false;
		}*/
		if(polygon.toList().size() < 6) {
			if(noisy) {
				System.out.println("Failed imposter test.");
			}
			return -1;
		}
		Moments m = Imgproc.moments(polygon, false);
		int y = (int) (m.get_m01() / m.get_m00());
		if(y < HEIGHT / 2) {
			return 6;
		}
		return 8;
	}

	public boolean TrackPolygonsInFrame(Mat mat) {
		if(!vc.isOpened()) {
			return false;
    }

		if(!vc.read(mat)) {
			return false;
		}
		
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat gray = new Mat();
		Mat gaussian = new Mat();
		Mat canny = new Mat();
		// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=HoughLinesP
		Imgproc.resize(mat, mat, new Size(WIDTH, HEIGHT), 0, 0); 
		Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
		Imgproc.GaussianBlur(gray, gaussian, new Size(GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), GAUSSIAN_X_STDDEV);
		Imgproc.Canny(gaussian, canny, 10, 160);
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
		Imgproc.dilate(canny, canny, element);
		Imgproc.erode(canny, canny, element);
		Imgproc.findContours(canny, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

		// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=arcLength
		Iterator<MatOfPoint> iterator = contours.iterator();
		List<Pair<Double, MatOfPoint>> hexagons = new ArrayList<Pair<Double, MatOfPoint>>();
		List<Pair<Double, MatOfPoint>> octagons = new ArrayList<Pair<Double, MatOfPoint>>();

		MatOfPoint largestHexagon = null;
		double largestHexagonArea = -1;

		while(iterator.hasNext()) {
			MatOfPoint contour = iterator.next();
			double epsilon = 0.005*Imgproc.arcLength(new MatOfPoint2f(contour.toArray()),true);
			MatOfPoint2f polygon2f = new MatOfPoint2f();
			Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()),polygon2f,epsilon,true);
			//Imgproc.approxPolyDP(polygon2f,polygon2f,0.01*Imgproc.arcLength(polygon2f,true),true);
			// https://stackoverflow.com/questions/30134903/convert-matofpoint2f-to-matofpoint
			MatOfPoint polygon = new MatOfPoint();
			polygon2f.convertTo(polygon, CvType.CV_32S);
			int result = polygonTests(polygon, false);
			if(result == -1) { continue; }
			else if(result == 6) {
				Pair<Double, MatOfPoint> hexagonPair = processHexagon(polygon);
				if(!hexagonTest(hexagonPair.getValue())) { continue; }
				if(!Imgproc.isContourConvex(hexagonPair.getValue())) { continue; }
				hexagons.add(hexagonPair);
			}
			else if(result == 8) {
				Pair<Double, MatOfPoint> octagonPair = processOctagon(polygon);
				if(octagonPair != null) {
					if(!Imgproc.isContourConvex(octagonPair.getValue())) { continue; }
					octagons.add(octagonPair);
				}
			}
		}

		Collections.sort(hexagons, new Comparator<Pair<Double, MatOfPoint>>() {
			@Override public int compare(final Pair<Double, MatOfPoint> p1, final Pair<Double, MatOfPoint> p2) {
				return Double.compare(p2.getKey(), p1.getKey());
			}
		});

		Collections.sort(octagons, new Comparator<Pair<Double, MatOfPoint>>() {
			@Override public int compare(final Pair<Double, MatOfPoint> p1, final Pair<Double, MatOfPoint> p2) {
				return Double.compare(p2.getKey(), p1.getKey());
			}
		});

		List<MatOfPoint> hexagonsToDraw = new ArrayList<MatOfPoint>();
		List<MatOfPoint> octagonsToDraw = new ArrayList<MatOfPoint>();
		for(int i=0;i<hexagons.size();i++) { hexagonsToDraw.add(hexagons.get(i).getValue()); }
		for(int i=0;i<octagons.size();i++) { octagonsToDraw.add(octagons.get(i).getValue()); }

		for(int i=0;i<Math.min(1, hexagonsToDraw.size());i++) {
			Imgproc.drawContours(mat, hexagonsToDraw, i, new Scalar(0, 255, 0), 1);
			for(int j=0;j<hexagonsToDraw.get(i).toList().size();j++) {
				Imgproc.putText(mat, Integer.toString(j+1), hexagonsToDraw.get(i).toList().get(j), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0));
			}
		}
		for(int i=0;i<Math.min(1, octagonsToDraw.size());i++) {
			Imgproc.drawContours(mat, octagonsToDraw, i, new Scalar(0, 0, 255), 1);
		}

		currHexagons = hexagonsToDraw;
		currOctagons = octagonsToDraw;

		panel.setImage(PolygonTracker.bufferedImage(mat));
		panel.repaint();
		return true;
	}

	public List<MatOfPoint> getFrameHexagons() {
		return currHexagons;
	}

	public List<MatOfPoint> getFrameOctagons() {
		return currOctagons;
	}
}

public class Test {

	public static int nFrames = 0;

	public static void writeCalibrationData(String filename, Mat cameraMatrix, Mat distCoeffs) {
		try {
			FileWriter fileWriter = new FileWriter(filename);
			fileWriter.write(Integer.toString(distCoeffs.rows()) + " " + Integer.toString(cameraMatrix.cols()) + "\r\n");
			for(int r=0;r<cameraMatrix.rows();r++) {
				for(int c=0;c<cameraMatrix.cols();c++) {
					for(int i=0;i<cameraMatrix.get(r, c).length;i++) {
						fileWriter.write(Double.toString(cameraMatrix.get(r, c)[i]) + " ");
					}
				}
				fileWriter.write("\r\n");
			}
			fileWriter.write(Integer.toString(distCoeffs.rows()) + " " + Integer.toString(cameraMatrix.cols()) + "\r\n");
			for(int r=0;r<distCoeffs.rows();r++) {
				for(int c=0;c<distCoeffs.cols();c++) {
					for(int i=0;i<distCoeffs.get(r, c).length;i++) {
						fileWriter.write(Double.toString(distCoeffs.get(r, c)[i]) + " ");
					}
				}
				fileWriter.write("\r\n");
			}
			fileWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void readCalibrationData(String filename, Mat cameraMatrix, Mat distCoeffs) {
		try {
			BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
			String line;
			int currentRow = -1;
			int maxRows = 0;
			boolean usingCameraMatrix = true;
			while((line = bufferedReader.readLine()) != null) {
				if(currentRow == -1) {
					String[] parts = line.split(" ", -1);
					maxRows = Integer.parseInt(parts[1]);
					currentRow++;
				}
				else {
					String[] parts = line.split(" ", -1);
					double[] d = new double[parts.length];
					for(int i=0;i<parts.length;i++) {
						if(parts[i].length() > 0) {
							d[i] = Double.parseDouble(parts[i]);
						}
					}
					if(usingCameraMatrix) {
						cameraMatrix.put(0, currentRow, d);
					}
					else {
						distCoeffs.put(0, currentRow, d);
					}
					currentRow++;
					if(currentRow == maxRows) { currentRow = -1; usingCameraMatrix = false; }
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void ComputeCalibration(String filename, Mat cameraMatrix, Mat distCoeffs) {
		List<Mat> savedHexagons = new ArrayList<Mat>();
		List<Mat> frames = new ArrayList<Mat>();

		PolygonTracker tracker = new PolygonTracker(filename + ".mov", 0);
		Mat mat = new Mat();
		int nFrames = 0;
		while(tracker.TrackPolygonsInFrame(mat)) {
			List<MatOfPoint> hexagons = tracker.getFrameHexagons();
			if (hexagons.size() == 0) {
				continue;
			}
		  nFrames++;
			if (nFrames > 400 && nFrames % 100 == 0) {
				savedHexagons.add(new MatOfPoint2f(hexagons.get(0).toArray()));
				frames.add(mat.clone());
				//Imgproc.putText(mat, "Calibrating!!!!!!!!!!!!!!!!!!", new org.opencv.core.Point(100, 100), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 127, 255), 10);
				tracker.panel.setImage(tracker.bufferedImage(mat));
				tracker.panel.repaint();
			}
			/*if (nFrames > 1100) {
				break;
			}*/
		}

		HexagonObject hexagonObject = new HexagonObject();

		List<Mat> objectPoints = new ArrayList<Mat>();
		for(int i=0;i<savedHexagons.size();i++) {
			objectPoints.add(hexagonObject.getObjectPoints());
		}

		List<Mat> rvecs = new ArrayList<>();
		List<Mat> tvecs = new ArrayList<>();
		Calib3d.calibrateCamera(objectPoints, savedHexagons, new Size(PolygonTracker.WIDTH, PolygonTracker.HEIGHT), cameraMatrix, distCoeffs, rvecs, tvecs);
		//writeCalibrationData(filename + "-calib.txt", cameraMatrix, distCoeffs);
  }

	public static void main(String[] args) {
		String videoFilename = "res/life-size-target-video";

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat cameraMatrix = new Mat();
	  Mat distCoeffs = new Mat();
		// https://howtodoinjava.com/java/io/how-to-check-if-file-exists-in-java/
		/*if(new File(videoFilename + "-calib.txt").exists()) {
			readCalibrationData(videoFilename + "-calib.txt", cameraMatrix, distCoeffs);
		}
		else {*/
		ComputeCalibration(videoFilename, cameraMatrix, distCoeffs);
		//}

		HexagonObject hexagonObject = new HexagonObject();
		PolygonTracker tracker = new PolygonTracker(videoFilename + ".mov", 0);

		Mat mat = new Mat();
		Mat rvec = new Mat();
		Mat tvec = new Mat();
		boolean useExtrinsicGuess = false;
		while(tracker.TrackPolygonsInFrame(mat)) {
			List<MatOfPoint> hexagons = tracker.getFrameHexagons();
			//List<MatOfPoint> octagons = tracker.getFrameOctagons();
			if (hexagons.size() == 0) {
				useExtrinsicGuess = false;
				continue;
			}
			MatOfPoint2f imagePoints = new MatOfPoint2f(hexagons.get(0).toArray());
			Calib3d.solvePnP(hexagonObject.getObjectPoints(), imagePoints, cameraMatrix, new MatOfDouble(distCoeffs), rvec, tvec, useExtrinsicGuess);
			useExtrinsicGuess = true;	
			
			// Draw points.
			MatOfPoint2f dMat1 = new MatOfPoint2f();
			Calib3d.projectPoints(
					hexagonObject.getObjectPoints(),
					rvec,
					tvec,
					cameraMatrix,
					new MatOfDouble(distCoeffs),
					dMat1
					);
			
			// Draw points.
			MatOfPoint2f dMat2 = new MatOfPoint2f();
			Calib3d.projectPoints(
					hexagonObject.getObjectPoints(2),
					rvec,
					tvec,
					cameraMatrix,
					new MatOfDouble(distCoeffs),
					dMat2
					);

			List<MatOfPoint> draw = new ArrayList<MatOfPoint>();

			MatOfPoint temp1 = new MatOfPoint();
			dMat1.convertTo(temp1, CvType.CV_32S);
			draw.add(temp1);

			MatOfPoint temp2 = new MatOfPoint();
			dMat2.convertTo(temp2, CvType.CV_32S);
			draw.add(temp2);

			Imgproc.drawContours(mat, draw, 0, new Scalar(0, 0, 255), 4);
			Imgproc.drawContours(mat, draw, 1, new Scalar(0, 0, 255), 4);
			for(int i=0;i<temp1.toList().size();i++) {
				Imgproc.line(mat, temp1.toList().get(i), temp2.toList().get(i), new Scalar(0, 0, 255), 4);
			}

			tracker.panel.setImage(PolygonTracker.bufferedImage(mat));
			tracker.panel.repaint();
		}
	}
}

class TestPanel extends JPanel {
	public BufferedImage image;

	public TestPanel() { this.image = null; }

	public void setImage(BufferedImage image) { this.image = image; }

	@Override public void paintComponent(Graphics g) {
		super.paintComponent(g);
		if(this.image != null) {
			g.drawImage(this.image, 0, 0, null);
		}
	}
}
