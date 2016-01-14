using System;
using System.Collections.Concurrent;
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
using Emgu.CV.XFeatures2D;

namespace GestureDetection.Extensions
{
    public static class MatExtensions
    {
        public class FixedSizedQueue<T>
        {
            public ConcurrentQueue<T> q = new ConcurrentQueue<T>();

            public int Limit { get; set; }
            public void Enqueue(T obj)
            {
                q.Enqueue(obj);
                lock (this)
                {
                    T overflow;
                    while (q.Count > Limit && q.TryDequeue(out overflow)) ;
                }
            }
        }

        private enum Directions
        {
            None,
            East,
            West,
            North,
            South
        }

        private const int MaxSize = 25;
        private static string Direction = "Move your hand";
        public static FixedSizedQueue<Point> QueueOfCentroids =
            new FixedSizedQueue<Point>() { Limit = MaxSize };

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
            var withContures = new Mat(frame.Size, DepthType.Cv8U, 3);

            using (var contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(frame, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                double largestCountourSize = 0;
                int largestCountourIndex = -1;
                for (int i = 0; i < contours.Size; i++)
                {
                    double a = CvInvoke.ContourArea(contours[i], false);  //  Find the area of contour
                    if (a > largestCountourSize)
                    {
                        largestCountourSize = a;
                        largestCountourIndex = i;                //Store the index of largest contour
                    }

                }
                CvInvoke.DrawContours(withContures, contours, largestCountourIndex, new MCvScalar(255, 255, 255), 5);

                if (largestCountourIndex > -1)
                {

                    ConvexityDefectsAndConvexHull(contours[largestCountourIndex], withContures);
                }

                //CvInvoke.ConvexityDefects(contours, convexHullPoints, withCorners);
                //CvInvoke.CornerHarris(withContures, withCorners, 3);
                //CvInvoke.Dilate(withCorners, afterDilatation, element, new Point(1, 1), 1, BorderType.Constant, new MCvScalar());
                //ValueType localMax = CvInvoke.cvCreateMat(afterDilatation.Height, afterDilatation.Width, DepthType.Cv8U);
                //CvInvoke.Compare(withCorners, afterDilatation, afterCompare, CmpType.Equal);


            }
            return withContures;
        }

        public static void ConvexityDefectsAndConvexHull(VectorOfPoint contour, Mat withContures)
        {
            using (var convexHull = new VectorOfInt())
            using (Mat convexityDefect = new Mat())
            {
                //Draw the contour in white thick line
                CvInvoke.ConvexHull(contour, convexHull);
                CvInvoke.ConvexityDefects(contour, convexHull, convexityDefect);

                if (!convexityDefect.IsEmpty)
                {
                    //Data from Mat are not directly readable so we convert it to Matrix<>
                    Matrix<int> m = new Matrix<int>(convexityDefect.Rows, convexityDefect.Cols,
                       convexityDefect.NumberOfChannels);
                    convexityDefect.CopyTo(m);

                    Point? lastPoint2 = null;
                    var vector = new VectorOfPoint();
                    var allPoints = new VectorOfPoint();
                    bool ignore = false;
                    for (int i = 0; i < m.Rows; i++)
                    {
                        var pointStart = contour.ToArray()[m.Data[i, 0]];
                        var pointEnd = contour.ToArray()[m.Data[i, 1]];
                        var pointFarthest = contour.ToArray()[m.Data[i, 2]];

                        var degreeAcos = Math.Acos((Math.Pow(CalculateDistance(pointStart, pointFarthest), 2)
                                                   + Math.Pow(CalculateDistance(pointEnd, pointFarthest), 2)
                                                   - Math.Pow(CalculateDistance(pointStart, pointEnd), 2))
                                               / (2 * CalculateDistance(pointStart, pointFarthest) * CalculateDistance(pointEnd, pointFarthest)));



                        CvInvoke.Polylines(withContures, new[] { pointStart, pointEnd }, true, new MCvScalar(0, 255, 255), 5);


                        vector.Push(new[] { pointStart, });
                        var degree = degreeAcos * 180 / Math.PI;
                        if (m.Data[i, 3] > 80 * 256 && degree > 25)
                        {
                            CvInvoke.Circle(withContures, Point.Round(pointFarthest), 7, new MCvScalar(255, 255, 0), 10);

                            allPoints.Push(new[] { pointStart, pointFarthest, pointEnd });
                        }
                    }

                    var orderedPoints = allPoints.ToArray().Distinct().OrderBy(x => Math.Sqrt(Math.Pow(x.X, 2) + Math.Pow(x.Y, 2)));
                    Point? lastPoint = null;

                    foreach (var orderedPoint in orderedPoints)
                    {
                        if (lastPoint.HasValue)
                        {
                            var length = Math.Sqrt(Math.Pow(lastPoint.Value.X - orderedPoint.X, 2) + Math.Pow(lastPoint.Value.Y - orderedPoint.Y, 2));

                            if (length < 50)
                                ignore = true;
                            else
                            {
                                ignore = false;

                            }
                        }

                        lastPoint = orderedPoint;
                        //if (!ignore)
                        //CvInvoke.Circle(withContures, Point.Round(orderedPoint), 3, new MCvScalar(0, 255, 0), 10);

                    }

                    CvInvoke.Polylines(withContures, vector, true, new MCvScalar(0, 0, 255), 2);

                    QueueOfCentroids.Enqueue(Compute2DPolygonCentroid(contour));
//                    if (QueueOfCentroids.q.Count == MaxSize)
//                    {
//                        var beginning = QueueOfCentroids.q.ElementAt(MaxSize - 5);
//                        var end = QueueOfCentroids.q.Last();
//                        var dx = QueueOfCentroids.q.ElementAt(MaxSize - 5).X - end.X;
//                        var dy = end.Y - QueueOfCentroids.q.ElementAt(MaxSize - 5).Y;
//
//                        Directions eastWest;
//                        Directions northSouth;
//
//                        SetDirections(dx, dy, out eastWest, out northSouth);
//                        ResolveDirection(eastWest, northSouth);
//
//                        CvInvoke.PutText(
//                            withContures,
//                            Direction,
//                            new Point(50, 50),
//                            FontFace.HersheyComplex,
//                            1,
//                            new MCvScalar(255, 255, 255),
//                            2);
//                        CvInvoke.Line(withContures, beginning, end, new MCvScalar(0, 0, 255), 5);
//                    }
                    foreach (var centroid in QueueOfCentroids.q)
                    {
                        CvInvoke.Circle(withContures, centroid, 4, new MCvScalar(0, 0, 255), 5);
                    }
                }
            }

        }

        private static void ResolveDirection(Directions eastWest, Directions northSouth)
        {
            if (eastWest == Directions.East)
            {
                if (northSouth == Directions.North)
                {
                    Direction = string.Format("North East");
                }
                else if (northSouth == Directions.South)
                {
                    Direction = string.Format("South East");
                }
                else
                {
                    Direction = string.Format("East");
                }
            }
            else if (eastWest == Directions.West)
            {
                if (northSouth == Directions.North)
                {
                    Direction = string.Format("North West");
                }
                else if (northSouth == Directions.South)
                {
                    Direction = string.Format("South West");
                }
                else
                {
                    Direction = string.Format("West");
                }
            }
        }

        private static void SetDirections(int dx, int dy, out Directions eastWest, out Directions northSouth)
        {
            Directions localEastWest = new Directions();
            Directions localNorthSouth = new Directions();

            if (Math.Abs(dx) > 10)
            {
                if (Math.Sign(dx) == 1)
                {
                    localEastWest = Directions.East;
                }
                else
                {
                    localEastWest = Directions.West;
                }
            }

            if (Math.Abs(dy) > 10)
            {
                if (Math.Sign(dx) == 1)
                {
                    localNorthSouth = Directions.North;
                }
                else
                {
                    localNorthSouth = Directions.South;
                }
            }

            eastWest = localEastWest;
            northSouth = localNorthSouth;
        }

        public static double CalculateDistance(Point p1, Point p2)
        {
            return Math.Sqrt(Math.Pow((p2.X - p1.X), 2) + Math.Pow((p2.Y - p1.Y), 2));
        }

        public static Point Compute2DPolygonCentroid(VectorOfPoint contour)
        {
            Point centroid = new Point() { X = 0, Y = 0 };
            int signedArea = 0;
            int x0; // Current vertex X
            int y0; // Current vertex Y
            int x1; // Next vertex X
            int y1; // Next vertex Y
            int a;  // Partial signed area

            // For all vertices except last
            int i;
            for (i = 0; i < contour.Size - 1; ++i)
            {
                x0 = contour[i].X;
                y0 = contour[i].Y;
                x1 = contour[i + 1].X;
                y1 = contour[i + 1].Y;
                a = x0 * y1 - x1 * y0;
                signedArea += a;
                centroid.X += (x0 + x1) * a;
                centroid.Y += (y0 + y1) * a;
            }

            // Do last vertex
            x0 = contour[i].X;
            y0 = contour[i].Y;
            x1 = contour[0].X;
            y1 = contour[0].Y;
            a = x0 * y1 - x1 * y0;
            signedArea += a;
            centroid.X += (x0 + x1) * a;
            centroid.Y += (y0 + y1) * a;

            signedArea = (int)(signedArea * 0.5);
            centroid.X = (int)(centroid.X / (6 * signedArea));
            centroid.Y = (int)(centroid.Y / (6 * signedArea));

            return centroid;
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

        public static int GetLowerLimitThreshold(this Mat frame)
        {
            var result = new Mat();
            int upperLimit = 200;
            int lowerLimit = 0;
            float fulfulfillmentWithBlack = 0;
            while (fulfulfillmentWithBlack < 0.9 && lowerLimit < upperLimit)
            {
                lowerLimit++;
                CvInvoke.Threshold(frame, result, lowerLimit, upperLimit, ThresholdType.Binary);
                float nonBlackPoints = CvInvoke.CountNonZero(result.ToGrey());
                float numbOfAllPoints = result.Width * result.Height;
                fulfulfillmentWithBlack = 1 - (nonBlackPoints / numbOfAllPoints);
            }
            return lowerLimit;
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
