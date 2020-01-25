import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import javax.swing.*;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;

public class Test {

	public static final String TITLE = "Test";
	public static final int WIDTH = 840;
	public static final int HEIGHT = 680;

	// TODO: change parameters for optimal performance
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
    if ( m.channels() > 1 ) {
			type = BufferedImage.TYPE_3BYTE_BGR;
    }
    BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
    m.get(0,0,((DataBufferByte)image.getRaster().getDataBuffer()).getData()); // get all the pixels
    return image;
	}

	public static double otsu(Mat mat) {
		// https://stackoverflow.com/questions/31289895/threshold-image-using-opencv-java
		// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=threshold
		Mat dummy = new Mat();
		return Imgproc.threshold(mat, dummy, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
	}

	public static void main(String[] args) {
		JFrame jframe = new JFrame(TITLE);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setSize(WIDTH, HEIGHT);

		TestPanel panel = new TestPanel();
		jframe.add(panel);

		jframe.setVisible(true);

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		VideoCapture videoCapture = new VideoCapture(0);
		Mat mat = new Mat();
		while(true) {
			if(videoCapture.isOpened()) {
				if(videoCapture.read(mat)) {
					List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
					Mat gray = new Mat();
					Mat gaussian = new Mat();
					Mat canny = new Mat();
					// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=HoughLinesP
					Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
					Imgproc.GaussianBlur(gray, gaussian, new Size(GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), GAUSSIAN_X_STDDEV);
					//double highThresh = otsu(gray);
					//double lowThresh = highThresh * 0.5;
					Imgproc.Canny(gaussian, canny, 10, 160);
					Mat hierarchy = new Mat();
					Imgproc.findContours(canny, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
					// https://stackoverflow.com/questions/30412592/opencv-drawcontours-in-java
					/*for (int i = 0; i < contours.size(); i++) {
						Imgproc.drawContours(gray, contours, i, new Scalar(255, 255, 255), -1, 8, hierarchy, 1);
					}*/

					// https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=arcLength
					Iterator<MatOfPoint> iterator = contours.iterator();
					List<MatOfPoint> temps = new ArrayList<MatOfPoint>();
					List<MatOfPoint2f> approximations = new ArrayList<MatOfPoint2f>();
					while(iterator.hasNext()) {
						MatOfPoint contour = iterator.next();
						double epsilon = 0.01*Imgproc.arcLength(new MatOfPoint2f(contour.toArray()),true);
						MatOfPoint2f approx = new MatOfPoint2f();
						Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()),approx,epsilon,true);
						approximations.add(approx);
						// https://stackoverflow.com/questions/30134903/convert-matofpoint2f-to-matofpoint
						MatOfPoint temp = new MatOfPoint();
						approx.convertTo(temp, CvType.CV_32S);
						temps.add(temp);
					}
					// https://stackoverflow.com/questions/30412592/opencv-drawcontours-in-java
					for(int i=0;i<temps.size();i++) {
						if(approximations.get(i).toList().size() == 6 || approximations.get(i).toList().size() == 8) {
							Imgproc.drawContours(mat, temps, i, new Scalar(255, 255, 255), -1);
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

	public TestPanel() {
		this.image = null;
	}

	public void setImage(BufferedImage image) {
		this.image = image;
	}

	@Override public void paintComponent(Graphics g) {
		super.paintComponent(g);
		if(this.image != null) {
			g.drawImage(this.image, 0, 0, null);
		}
	}
}
