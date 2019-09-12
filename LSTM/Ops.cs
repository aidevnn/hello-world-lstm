using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace LSTMsharp
{
    public static class Ops
    {
        public static Random Random = new Random();

        public static string Glue<T>(this IEnumerable<T> l, string sep = " ", string fmt = "{0}") => string.Join(sep, l.Select(v => string.Format(fmt, v)).ToArray());

        public static string Print(double[][] a, string fmt = "{0}")
        {
            StringBuilder sb = new StringBuilder();
            int m = a.Length;

            for(int k = 0; k < m; ++k)
            {
                string p = k == 0 ? "[[" : " [";
                string e = k == m - 1 ? "]]" : "]";
                sb.AppendLine($"{p}{a[k].Glue(", ", fmt)}{e}");
            }

            return sb.ToString();
        }

        public static double[][] Zeros(int m, int n)
        {
            double[][] r = new double[m][];
            for (int k = 0; k < m; ++k)
                r[k] = new double[n];

            return r;
        }

        public static double[][] Zeros(double[][] w)
        {
            int m = w.Length;
            int n = w[0].Length;
            return Zeros(m, n);
        }

        public static double[][] Randn(int m, int n, double std, double init)
        {
            double[][] r = new double[m][];
            for (int i = 0; i < m; ++i)
                r[i] = Enumerable.Range(0, n).Select(a => (10.0 * Random.NextDouble() - 5.0) * std + init).ToArray();

            return r;
        }


        public static double[][] Randint(int m, int n, int min, int max)
        {
            double[][] r = new double[m][];
            for (int i = 0; i < m; ++i)
                r[i] = Enumerable.Range(0, n).Select(a => (double)Random.Next(min, max)).ToArray();

            return r;
        }

        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double[][] Sigmoid(double[][] x) => x.Select(r => r.Select(Sigmoid).ToArray()).ToArray();

        public static double DSigmoid(double y) => y * (1.0 - y);
        public static double[][] DSigmoid(double[][] y) => y.Select(r => r.Select(DSigmoid).ToArray()).ToArray();

        public static double Tanh(double x) => Math.Tanh(x);
        public static double[][] Tanh(double[][] x) => x.Select(r => r.Select(Tanh).ToArray()).ToArray();

        public static double DTanh(double y) => 1.0 - y * y;
        public static double[][] DTanh(double[][] y) => y.Select(r => r.Select(DTanh).ToArray()).ToArray();

        public static double[][] Exp(double[][] x) => x.Select(r => r.Select(Math.Exp).ToArray()).ToArray();
        public static double[][] Sqrt(double[][] x, double eps = 1e-8) => x.Select(r => r.Select(c => Math.Sqrt(c + eps)).ToArray()).ToArray();
        public static double[][] Mul(double[][] x, double c0) => x.Select(r => r.Select(c => c * c0).ToArray()).ToArray();
        public static double[][] Div(double[][] x, double c0) => x.Select(r => r.Select(c => c / c0).ToArray()).ToArray();

        public static double Min(double[][] x) => x.SelectMany(a => a).Min();

        public static double[][] RowStack(double[][] a,double[][] b)
        {
            int m0 = a.Length;
            int m1 = b.Length;
            double[][] r = new double[m0 + m1][];

            for (int i = 0; i < m0; ++i)
                r[i] = a[i].ToArray();

            for (int i = 0; i < m1; ++i)
                r[i + m0] = b[i].ToArray();

            return r;
        }

        public static double[][] Dot(double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (na != mb) throw new Exception();

            double[][] r = new double[ma][];
            for(int i = 0; i < ma; ++i)
            {
                r[i] = new double[nb];
                for(int j = 0; j < nb; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < na; ++k)
                        sum += a[i][k] * b[k][j];

                    r[i][j] = sum;
                }
            }

            return r;
        }

        public static double[][] Add(double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (ma != mb || na != nb) throw new Exception();

            double[][] r = new double[ma][];

            for (int i = 0; i < ma; ++i)
            {
                r[i] = new double[na];
                for (int j = 0; j < na; ++j)
                    r[i][j] = a[i][j] + b[i][j];
            }

            return r;
        }

        public static void AddRef(double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (ma != mb || na != nb) throw new Exception();

            for (int i = 0; i < ma; ++i)
            {
                for (int j = 0; j < na; ++j)
                    a[i][j] += b[i][j];
            }
        }

        public static void AddRef2(ref double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (ma != mb || na != nb) throw new Exception();

            for (int i = 0; i < ma; ++i)
            {
                for (int j = 0; j < na; ++j)
                    a[i][j] += b[i][j];
            }
        }

        public static double[][] Mul(double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (ma != mb || na != nb) throw new Exception();

            double[][] r = new double[ma][];

            for (int i = 0; i < ma; ++i)
            {
                r[i] = new double[na];
                for (int j = 0; j < na; ++j)
                    r[i][j] = a[i][j] * b[i][j];
            }

            return r;
        }

        public static double[][] Div(double[][] a, double[][] b)
        {
            int ma = a.Length;
            int mb = b.Length;
            int na = a[0].Length;
            int nb = b[0].Length;

            if (ma != mb || na != nb) throw new Exception();

            double[][] r = new double[ma][];

            for (int i = 0; i < ma; ++i)
            {
                r[i] = new double[na];
                for (int j = 0; j < na; ++j)
                    r[i][j] = a[i][j] / b[i][j];
            }

            return r;
        }

        public static double Sum(double[][] a) => a.SelectMany(r => r).Sum();

        public static double[][] Copy(double[][] a) => Mul(a, 1.0);

        public static double[][] T(double[][] a)
        {
            int m = a.Length;
            int n = a[0].Length;

            double[][] r = new double[n][];

            for (int i = 0; i < n; ++i)
            {
                r[i] = new double[m];
                for (int j = 0; j < m; ++j)
                    r[i][j] = a[j][i];
            }

            return r;
        }

        public static void Clip(double[][] a, double min, double max)
        {
            int m = a.Length;
            int n = a[0].Length;

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                    a[i][j] = Math.Min(max, Math.Max(min, a[i][j]));
            }
        }

        public static void Fill(double[][] a, double v)
        {
            int m = a.Length;
            int n = a[0].Length;

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                    a[i][j] = v;
            }
        }

        public static double[][] Subarray(double[][] a, int hIdx)
        {
            int m = a.Length;
            int n = a[0].Length;

            if (hIdx > m) throw new Exception();

            double[][] r = new double[hIdx][];
            for (int i = 0; i < hIdx; ++i)
                r[i] = a[i].ToArray();

            return r;

        }

        public static void Assert(int[] a, int t)
        {
            if (a.Length != t) throw new Exception();
        }

        public static void Assert(double[][] a, (int, int) t)
        {
            int m = a.Length;
            int n = a[0].Length;

            if (m != t.Item1 || n != t.Item2) throw new Exception();
        }

        public static int[] Range(int a) => Enumerable.Range(0, a).ToArray();

        public static int Choice(int[] idx, double[] p)
        {
            if (p.Length >= 1 && p.Length != idx.Length) throw new Exception();

            double tot = p.Sum();
            var p0 = p.Select(v => v / tot).ToArray();

            double r = Random.NextDouble();
            double sum = 0;

            for(int k = 0; k < idx.Length; ++k)
            {
                sum += p0[k];
                if (r < sum)
                    return idx[k];
            }

            return idx.Last();
        }

        public static int[] Sequence(int[] arr, int start, int length) => arr.Skip(start).Take(length).ToArray();

        public static double[][] ARange(int m, int n, int start)
        {
            var r = Zeros(m, n);

            for(int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                    r[i][j] = start++;
            }

            return r;
        }
    }
}
