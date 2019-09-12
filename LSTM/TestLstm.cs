using System;
using System.Collections.Generic;

namespace LSTMsharp
{
    public class TestLstm
    {
        LSTM lstm;
        int xs = 2, hs = 3, ts = 3;
        int zs;
        public TestLstm()
        {
            zs = xs + hs;
            lstm = new LSTM(xs, hs, ts);

            lstm.W_f = Ops.ARange(hs, zs, 0);
            lstm.b_f = Ops.ARange(hs, 1, 0);
            lstm.W_i = Ops.ARange(hs, zs, 1);
            lstm.b_i = Ops.ARange(hs, 1, 1);
            lstm.W_C = Ops.ARange(hs, zs, 2);
            lstm.b_C = Ops.ARange(hs, 1, 2);
            lstm.W_o = Ops.ARange(hs, zs, 3);
            lstm.b_o = Ops.ARange(hs, 1, 3);
            lstm.W_y = Ops.ARange(xs, hs, 4);
            lstm.b_y = Ops.ARange(xs, 1, 4);

            lstm.dW_f = Ops.ARange(hs, zs, -1);
            lstm.db_f = Ops.ARange(hs, 1, -1);
            lstm.dW_i = Ops.ARange(hs, zs, -2);
            lstm.db_i = Ops.ARange(hs, 1, -2);
            lstm.dW_C = Ops.ARange(hs, zs, -3);
            lstm.db_C = Ops.ARange(hs, 1, -3);
            lstm.dW_o = Ops.ARange(hs, zs, -4);
            lstm.db_o = Ops.ARange(hs, 1, -4);
            lstm.dW_y = Ops.ARange(xs, hs, -5);
            lstm.db_y = Ops.ARange(xs, 1, -5);

            lstm.mW_f = Ops.ARange(hs, zs, 1);
            lstm.mb_f = Ops.ARange(hs, 1, 1);
            lstm.mW_i = Ops.ARange(hs, zs, 2);
            lstm.mb_i = Ops.ARange(hs, 1, 2);
            lstm.mW_C = Ops.ARange(hs, zs, 3);
            lstm.mb_C = Ops.ARange(hs, 1, 3);
            lstm.mW_o = Ops.ARange(hs, zs, 4);
            lstm.mb_o = Ops.ARange(hs, 1, 4);
            lstm.mW_y = Ops.ARange(xs, hs, 5);
            lstm.mb_y = Ops.ARange(xs, 1, 5);
        }

        public void Test1()
        {
            var g_h_prev = Ops.ARange(hs, 1, -6);
            var g_C_prev = Ops.ARange(hs, 1, -7);
            var x = Ops.Zeros(xs, 1);
            x[0][0] = 1.0;

            var (z0, f0, i0, C_bar0, C0, o0, h0, y0, p0) = lstm.Forward(x, g_h_prev, g_C_prev);
            var (dh_prev, dC_prev) = lstm.Backward(0, g_h_prev, g_C_prev, C0, z0, f0, i0, C_bar0, C0, o0, h0, y0, p0);

            Console.WriteLine("z");
            Console.WriteLine(Ops.Print(z0));
            Console.WriteLine();

            Console.WriteLine("f");
            Console.WriteLine(Ops.Print(f0));
            Console.WriteLine();

            Console.WriteLine("i");
            Console.WriteLine(Ops.Print(i0));
            Console.WriteLine();

            Console.WriteLine("C_bar");
            Console.WriteLine(Ops.Print(C_bar0));
            Console.WriteLine();

            Console.WriteLine("C");
            Console.WriteLine(Ops.Print(C0));
            Console.WriteLine();

            Console.WriteLine("o");
            Console.WriteLine(Ops.Print(o0));
            Console.WriteLine();

            Console.WriteLine("h");
            Console.WriteLine(Ops.Print(h0));
            Console.WriteLine();

            Console.WriteLine("y");
            Console.WriteLine(Ops.Print(y0));
            Console.WriteLine();

            Console.WriteLine("p");
            Console.WriteLine(Ops.Print(p0));
            Console.WriteLine();

            Console.WriteLine("dh_prev");
            Console.WriteLine(Ops.Print(dh_prev));
            Console.WriteLine();

            Console.WriteLine("dC_prev");
            Console.WriteLine(Ops.Print(dC_prev));
            Console.WriteLine();

        }

        public void Test2()
        {
            var g_h_prev = Ops.ARange(hs, 1, -6);
            var g_C_prev = Ops.ARange(hs, 1, -7);
            int[] inputs = { 0, 1, 0 };
            int[] targets = { 1, 0, 0 };

            var (loss, h_prev, C_prev) = lstm.ForwardBackward(inputs, targets, g_h_prev, g_C_prev);

            Console.WriteLine("loss");
            Console.WriteLine(loss);
            Console.WriteLine();

            Console.WriteLine("h_prev");
            Console.WriteLine(Ops.Print(h_prev));
            Console.WriteLine();

            Console.WriteLine("C_prev");
            Console.WriteLine(Ops.Print(C_prev));
            Console.WriteLine();
        }

        public void Test3()
        {
            lstm.g_h_prev = Ops.ARange(hs, 1, -6);
            lstm.g_C_prev = Ops.ARange(hs, 1, -7);
            int[] inputs = { 0, 1, 0 };
            int[] targets = { 1, 0, 0 };

            lstm.TrainOnBatch(inputs, targets);

            Console.WriteLine("loss");
            Console.WriteLine(lstm.smooth_loss);
            Console.WriteLine();

            Console.WriteLine("h_prev");
            Console.WriteLine(Ops.Print(lstm.g_h_prev));
            Console.WriteLine();

            Console.WriteLine("C_prev");
            Console.WriteLine(Ops.Print(lstm.g_C_prev));
            Console.WriteLine();

            Console.WriteLine("W_f");
            Console.WriteLine(Ops.Print(lstm.W_f));
            Console.WriteLine("W_i");
            Console.WriteLine(Ops.Print(lstm.W_i));
            Console.WriteLine("W_C");
            Console.WriteLine(Ops.Print(lstm.W_C));
            Console.WriteLine("W_o");
            Console.WriteLine(Ops.Print(lstm.W_o));
            Console.WriteLine("W_y");
            Console.WriteLine(Ops.Print(lstm.W_y));
            Console.WriteLine("b_f");
            Console.WriteLine(Ops.Print(lstm.b_f));
            Console.WriteLine("b_i");
            Console.WriteLine(Ops.Print(lstm.b_i));
            Console.WriteLine("b_C");
            Console.WriteLine(Ops.Print(lstm.b_C));
            Console.WriteLine("b_o");
            Console.WriteLine(Ops.Print(lstm.b_o));
            Console.WriteLine("b_y");
            Console.WriteLine(Ops.Print(lstm.b_y));

            Console.WriteLine("dW_f");
            Console.WriteLine(Ops.Print(lstm.dW_f));
            Console.WriteLine("dW_i");
            Console.WriteLine(Ops.Print(lstm.dW_i));
            Console.WriteLine("dW_C");
            Console.WriteLine(Ops.Print(lstm.dW_C));
            Console.WriteLine("dW_o");
            Console.WriteLine(Ops.Print(lstm.dW_o));
            Console.WriteLine("dW_y");
            Console.WriteLine(Ops.Print(lstm.dW_y));
            Console.WriteLine("db_f");
            Console.WriteLine(Ops.Print(lstm.db_f));
            Console.WriteLine("db_i");
            Console.WriteLine(Ops.Print(lstm.db_i));
            Console.WriteLine("db_C");
            Console.WriteLine(Ops.Print(lstm.db_C));
            Console.WriteLine("db_o");
            Console.WriteLine(Ops.Print(lstm.db_o));
            Console.WriteLine("db_y");
            Console.WriteLine(Ops.Print(lstm.db_y));

            Console.WriteLine("mW_f");
            Console.WriteLine(Ops.Print(lstm.mW_f));
            Console.WriteLine("mW_i");
            Console.WriteLine(Ops.Print(lstm.mW_i));
            Console.WriteLine("mW_C");
            Console.WriteLine(Ops.Print(lstm.mW_C));
            Console.WriteLine("mW_o");
            Console.WriteLine(Ops.Print(lstm.mW_o));
            Console.WriteLine("mW_y");
            Console.WriteLine(Ops.Print(lstm.mW_y));
            Console.WriteLine("mb_f");
            Console.WriteLine(Ops.Print(lstm.mb_f));
            Console.WriteLine("mb_i");
            Console.WriteLine(Ops.Print(lstm.mb_i));
            Console.WriteLine("mb_C");
            Console.WriteLine(Ops.Print(lstm.mb_C));
            Console.WriteLine("mb_o");
            Console.WriteLine(Ops.Print(lstm.mb_o));
            Console.WriteLine("mb_y");
            Console.WriteLine(Ops.Print(lstm.mb_y));


        }
    }
}
