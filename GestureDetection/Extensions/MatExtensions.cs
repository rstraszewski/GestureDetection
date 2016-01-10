using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.VideoSurveillance;

namespace GestureDetection.Extensions
{
    public static class MatExtensions
    {
        public static Mat Skeletonize(this Mat frame)
        {
            CvInvoke.Threshold(frame, frame, 127, 255, ThresholdType.Binary);
            var skel = new Mat(frame.Size, DepthType.Cv8U, 1);
            var temp = new Mat(frame.Size, DepthType.Cv8U, 1);
            var eroded = new Mat(frame.Size, DepthType.Cv8U, 1);
            skel.SetTo(new Bgr(0, 0, 0).MCvScalar);

            var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(1, 1));

            do
            {
                CvInvoke.Erode(frame, eroded, element, new Point(1, 1), 1, BorderType.Constant, new MCvScalar());
                CvInvoke.Dilate(eroded, temp, element, new Point(1, 1), 1, BorderType.Constant, new MCvScalar());
                CvInvoke.Subtract(frame, temp, temp);
                CvInvoke.BitwiseOr(skel, temp, skel);
                eroded.CopyTo(frame);
            } while (CvInvoke.CountNonZero(frame) != 0);

            return skel;
        }

        public static Mat ConvexHull(this Mat frame)
        {
            var withContures = new Mat(frame.Size, DepthType.Cv8U, 1);
            var withCorners = new Mat(frame.Size, DepthType.Cv8U, 1);
            var afterDilatation = new Mat(frame.Size, DepthType.Cv8U, 1);
            var afterCompare = new Mat(frame.Size, DepthType.Cv8U, 1);

            using (var convexHullPoints = new VectorOfPoint())
            using (var contours = new VectorOfVectorOfPoint())
            {
                var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(1, 1));

                CvInvoke.FindContours(frame, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                CvInvoke.DrawContours(withContures, contours, -1, new MCvScalar(255, 255, 255), 1);
                CvInvoke.ConvexHull(withContures, convexHullPoints);
                for (int i = 0; i < convexHullPoints.Size; i++)
                {
                    CvInvoke.Circle(withContures, convexHullPoints[i], 3, new MCvScalar(100, 100, 100));
                }
                //CvInvoke.ConvexityDefects(contours, convexHullPoints, withCorners);
                //CvInvoke.CornerHarris(withContures, withCorners, 3);
                //CvInvoke.Dilate(withCorners, afterDilatation, element, new Point(1, 1), 1, BorderType.Constant, new MCvScalar());
                //ValueType localMax = CvInvoke.cvCreateMat(afterDilatation.Height, afterDilatation.Width, DepthType.Cv8U);
                //CvInvoke.Compare(withCorners, afterDilatation, afterCompare, CmpType.Equal);


            }
            return withContures;
        }

        public static Mat SubtrackBackground(this IInputArrayOfArrays frame, BackgroundSubtractor backgroundSubtractor = null)
        {
            if (backgroundSubtractor == null)
            {
                backgroundSubtractor = new BackgroundSubtractorMOG2();
            }

            var output = new Mat();
            backgroundSubtractor.Apply(frame, output);

            return output;
        }
  
        public static Mat GaussianBlur(this Mat frame, Size size)
        {
            var result = new Mat();
            CvInvoke.GaussianBlur(frame, result, size, 0);

            return result;
        }

        public static Mat ToGrey(this Mat frame)
        {
            var result = new Mat();
            CvInvoke.CvtColor(frame, result, ColorConversion.Bgr2Gray);
            return result;
        }

        public static Mat Threshold(this Mat frame, double from, double to)
        {
            var result = new Mat();
            CvInvoke.Threshold(frame, result, from, to, ThresholdType.Binary);

            return result;
        }

        public static Mat Dilate(this Mat frame, int interations = 1, IInputArray element = null)
        {
            var result = new Mat();
            CvInvoke.Dilate(frame, result, element, new Point(1, 1), interations, BorderType.Constant, new MCvScalar());

            return result;
        }

        public static Mat Erode(this Mat frame, int interations = 1, IInputArray element = null)
        {
            var result = new Mat();
            CvInvoke.Erode(frame, result, element, new Point(1, 1), interations, BorderType.Constant, new MCvScalar());

            return result;
        }

        public static Mat AbsDiff(this Mat src1, Mat src2)
        {
            var result = new Mat();
            CvInvoke.AbsDiff(src1, src2, result);
            return result;
        }
    }
}
