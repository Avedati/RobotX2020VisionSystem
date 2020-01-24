import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Comparator;
import java.util.Collections;
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

public class Test {

	public static final String TITLE = "Test";
	public static final int WIDTH = 840;
	public static final int HEIGHT = 680;

	public static int GAUSSIAN_BLUR_KERNEL_SIZE = 7;
	public static double GAUSSIAN_X_STDDEV = 0;

	public static int nFrames = 0;

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

	public static Pair<Double, MatOfPoint> processHexagon(MatOfPoint polygon) {
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

		ArrayList<Integer> result = new ArrayList<Integer>();
		for(int i=0;i<6;i++) {
			result.add(pairs.get(i).getValue().intValue());
		}

		Collections.sort(result);

		ArrayList<org.opencv.core.Point> result2 = new ArrayList<org.opencv.core.Point>();
		for(int i=0;i<6;i++) {
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

	public static void main(String[] args) {
		JFrame jframe = new JFrame(TITLE);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setSize(WIDTH, HEIGHT);

		TestPanel panel = new TestPanel();
		jframe.add(panel);
		jframe.setVisible(true);

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		VideoCapture videoCapture = new VideoCapture("/Users/spiderfencer/Desktop/opencv-test-video.mov");
		videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, WIDTH);
		videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, HEIGHT);
		Mat mat = new Mat();

		List<MatOfPoint> savedHexagons = new ArrayList<MatOfPoint>();

		while(true) {
			if(videoCapture.isOpened()) {
				if(videoCapture.read(mat)) {
					Test.nFrames++;
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

					if(Test.nFrames % 100 == 0) {
						savedHexagons.add(hexagonsToDraw.get(0));
					}

					for(int i=0;i<Math.min(1, hexagonsToDraw.size());i++) {
						Imgproc.drawContours(mat, hexagonsToDraw, i, new Scalar(0, 255, 0), 1);
						for(int j=0;j<hexagonsToDraw.get(i).toList().size();j++) {
							Imgproc.putText(mat, Integer.toString(j+1), hexagonsToDraw.get(i).toList().get(j), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0));
						}
					}
					for(int i=0;i<Math.min(1, octagonsToDraw.size());i++) {
						Imgproc.drawContours(mat, octagonsToDraw, i, new Scalar(0, 0, 255), 1);
					}

					panel.setImage(Test.bufferedImage(mat));
					panel.repaint();
					if(Test.nFrames >= 1000) { break; }
					continue;
				}
			}
			break;
		}

		/*List<MatOfPoint> objectPoints = new ArrayList<MatOfPoint>();
		List<Point> pts = new ArrayList<Point>();
		// TODO: this
		for(double theta=0;theta<Math.PI*2;theta+=Math.PI/3) {
			pts.add(new Point(15 * Math.cos(theta), 15 * Math.cos(theta)));
		}
		for(int i=0;i<10;i++) {
			MatOfPoint mpt = new MatOfPoint();
			mpt.fromList(pts);
			objectPoints.add(mpt);
		}

		Mat cameraMatrix = new Mat();
		Mat distCoeffs = new Mat();
		List<Mat> rvecs = new ArrayList<>();
		List<Mat> tvecs = new ArrayList<>();
		Calib3d.calibrateCamera(objectPoints, savedHexagons, new Size(WIDTH, HEIGHT), cameraMatrix, distCoeffs, rvecs, tvecs);*/
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
