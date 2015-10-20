using Emgu.CV;
using System;
using System.Windows;
using System.Windows.Threading;
using Emgu.CV.BgSegm;
using Emgu.CV.CvEnum;
using Emgu.CV.Utility;
using Emgu.CV.VideoSurveillance;
using GestureDetection.Extensions;

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
            backgroundSubtractor = new BackgroundSubtractorGMG(360, 0.7);
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            Camera.Source = capture
                .QueryFrame()
                .SubtrackBackground(backgroundSubtractor)
                .GaussianBlur()
                .Threshold(175, 255)
                .Skeletonize()
                .ToBitmapSource();
        }

        
    }
}
