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

public class Test {

	public static final String TITLE = "Test";
	public static final int WIDTH = 840;
	public static final int HEIGHT = 680;

	public static int GAUSSIAN_BLUR_KERNEL_SIZE = 7;
	public static double GAUSSIAN_X_STDDEV = 1;
	public static double HOUGHLINESP_RHO = 1;
	public static double HOUGHLINESP_THETA = Math.PI / 180;
	public static double HOUGHLINESP_THRESHOLD = 100;
	public static double HOUGHLINESP_MINLINELENGTH = 50;
	public static double HOUGHLINESP_MAXLINEGAP = 5;

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

	public static boolean isHexagon(MatOfPoint polygon, boolean noisy) {
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
		
		if(!flag) {
			if(noisy) {
				System.out.println("Failed side length test.");
			}
			return false;
		}
		if(!Imgproc.isContourConvex(polygon)) {
			if(noisy) {
				System.out.println("Failed convexity test.");
			}
			return false;
		}
		if(polygon.toList().size() < 6) {
			if(noisy) {
				System.out.println("Failed hexagon test.");
			}
			return false;
		}
		if(noisy) {
			System.out.println("Succeeded!");
		}
		return true;
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
		videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 800);
		videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 600);
		Mat mat = new Mat();

		while(true) {
			if(videoCapture.isOpened()) {
				if(videoCapture.read(mat)) {
					List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
					Mat gray = new Mat();
					Mat gaussian = new Mat();
					Mat canny = new Mat();
					// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=HoughLinesP
					Imgproc.resize(mat, mat, new Size(800, 600), 0, 0); 
					Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
					Imgproc.GaussianBlur(gray, gaussian, new Size(GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), GAUSSIAN_X_STDDEV);
					Imgproc.Canny(gaussian, canny, 10, 160);
					Imgproc.findContours(canny, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

					// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=arcLength
					Iterator<MatOfPoint> iterator = contours.iterator();
					List<Pair<Double, MatOfPoint>> hexagons = new ArrayList<>();

					MatOfPoint largestHexagon = null;
					double largestHexagonArea = -1;

					while(iterator.hasNext()) {
						MatOfPoint contour = iterator.next();
						double epsilon = 0.005*Imgproc.arcLength(new MatOfPoint2f(contour.toArray()),true);
						MatOfPoint2f polygon2f = new MatOfPoint2f();
						Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()),polygon2f,epsilon,true);
						Imgproc.approxPolyDP(polygon2f,polygon2f,0.01*Imgproc.arcLength(polygon2f,true),true);
						// https://stackoverflow.com/questions/30134903/convert-matofpoint2f-to-matofpoint
						MatOfPoint polygon = new MatOfPoint();
						polygon2f.convertTo(polygon, CvType.CV_32S);
						if(!isHexagon(polygon, false)) { continue; }
						double area = Imgproc.contourArea(polygon);
						hexagons.add(new Pair(area, polygon));
						/*else if(approx.toList().size() == 8 && area > largestArea8) {
							largestArea8 = area;
							largestContour8 = temp;
						}*/
					}

					// https://stackoverflow.com/questions/29920027/how-can-i-sort-a-list-of-pairstring-integer
					Collections.sort(hexagons, new Comparator<Pair<Double, MatOfPoint>>() {
						@Override public int compare(final Pair<Double, MatOfPoint> o1, final Pair<Double, MatOfPoint> o2) {
							double d1 = o1.getKey().doubleValue();
							double d2 = o2.getKey().doubleValue();
							return (int)(d2 - d1);
						}
					});

					List<MatOfPoint> drawnHexagons = new ArrayList<>();
					for(int i=0;i<hexagons.size();i++) { drawnHexagons.add(hexagons.get(i).getValue()); }
					for(int i=0;i<Math.min(drawnHexagons.size(), 4);i++) {
						int numSides = drawnHexagons.get(i).toList().size();
						/*if(numSides == 7) {
							Imgproc.drawContours(mat, drawnHexagons, i, new Scalar(0, 255, 0), 4);
						}
						else if(numSides == 9) {
							Imgproc.drawContours(mat, drawnHexagons, i, new Scalar(0, 127, 255), 4);
						}
						else */if(numSides == 8) {
							Imgproc.drawContours(mat, drawnHexagons, i, new Scalar(255, 0, 0), 4);
						}
						else if(numSides == 6) {
							Imgproc.drawContours(mat, drawnHexagons, i, new Scalar(0, 0, 255), 4);
						}
						for(org.opencv.core.Point p: drawnHexagons.get(i).toList()) {
							Imgproc.circle(mat, p, 8, new Scalar(199, 110, 255), -1);
						}
					}

					panel.setImage(Test.bufferedImage(mat));
					panel.repaint();
					continue;
				}
			}
			break;
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
