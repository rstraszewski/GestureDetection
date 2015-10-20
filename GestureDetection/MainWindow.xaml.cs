using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Utility;
using Emgu.CV.VideoSurveillance;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;

namespace GestureDetection
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private Capture capture;
        private DispatcherTimer timer;
        private BackgroundSubtractor backgroundSubtractor;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            InitializeTimer();

            InitializeEmguCv();
        }

        private void InitializeTimer()
        {
            timer = new DispatcherTimer();
            timer.Tick += ProcessFrame;
            timer.Interval = new TimeSpan(0, 0, 0, 0, 16);
            timer.Start();
        }

        private void InitializeEmguCv()
        {
            capture = new Capture(CaptureType.Any);
            backgroundSubtractor = new BackgroundSubtractorMOG2();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            var frame = capture.QueryFrame();
            var subtracked = SubtrackBackground(frame);
            var skeletonized = Skeletonize(subtracked);
            Camera.Source = skeletonized.ToBitmapSource();
        }

        private Mat SubtrackBackground(IInputArrayOfArrays frame)
        {
            var output = new Mat();
            backgroundSubtractor.Apply(frame, output);

            return output;
        }

        private Mat Skeletonize(Mat frame)
        {
            CvInvoke.Threshold(frame, frame, 127, 255, ThresholdType.Binary);
            var skel = new Mat(frame.Size, DepthType.Cv8U, 1);
            var temp = new Mat(frame.Size, DepthType.Cv8U, 1);
            var eroded = new Mat(frame.Size, DepthType.Cv8U, 1);
            skel.SetTo(new Bgr(0,0,0).MCvScalar);

            var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(1,1));

            do
            {
                CvInvoke.Erode(frame, eroded, element, new Point(1,1), 1, BorderType.Constant, new MCvScalar());
                CvInvoke.Dilate(eroded, temp, element, new Point(1,1), 1, BorderType.Constant, new MCvScalar());
                CvInvoke.Subtract(frame, temp, temp);
                CvInvoke.BitwiseOr(skel, temp, skel);
                eroded.CopyTo(frame);
            } while (CvInvoke.CountNonZero(frame) != 0);

            return skel;
        }
    }
}
