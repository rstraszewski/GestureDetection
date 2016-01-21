using Emgu.CV;
using System;
using System.Drawing;
using System.Windows;
using System.Windows.Threading;
using Emgu.CV.BgSegm;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Utility;
using Emgu.CV.VideoSurveillance;
using GestureDetection.Extensions;
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
        private Mat background;
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

            background = capture
                .QueryFrame()
                .ToGrey()
                .GaussianBlur(new Size(11, 11));

        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            var frame = capture
                .QueryFrame();

           var lowerLimit = frame.GetLowerLimitThreshold();
           var mask = frame
                .ToGrey()
                .GaussianBlur(new Size(21, 21))
                .AbsDiff(background)
                .Threshold(lowerLimit, 200)
                .Dilate(2);
            int count = 0;
            Camera.Source = mask.ToBitmapSource();
            CameraConvexHull.Source = mask
                .ConvexHull(ref count)
                .ToBitmapSource();
            
        }

        
    }
}
