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
            Camera.Source = SubtrackBackground(frame).ToBitmapSource();
        }

        private Mat SubtrackBackground(IInputArrayOfArrays frame)
        {
            var output = new Mat();
            backgroundSubtractor.Apply(frame, output);

            return output;
        }
    }
}
