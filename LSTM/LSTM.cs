using System;
using System.Collections.Generic;
using System.Linq;

namespace LSTMsharp
{
    public class LSTM
    {
        public LSTM(int x_size, int h_size, int t_steps)
        {
            X_size = x_size;
            H_size = h_size;
            T_steps = t_steps;
            z_size = H_size + X_size;

            Init();
        }

        public int X_size = 10; // Size of the input layer
        public int H_size = 100; // Size of the hidden layer
        public int T_steps = 25; // Number of time steps (length of the sequence) used for training
        public double learning_rate = 1e-1; // Learning rate
        public double weight_sd = 0.1; // Standard deviation of weights for initialization
        public double smooth_loss; // Exponential average of loss. 
        public int z_size; // Size of concatenate(H, X) vector

        // weights
        public double[][] W_f, b_f;
        public double[][] W_i, b_i;
        public double[][] W_C, b_C;
        public double[][] W_o, b_o;
        public double[][] W_y, b_y;

        // gradients
        public double[][] dW_f, db_f;
        public double[][] dW_i, db_i;
        public double[][] dW_C, db_C;
        public double[][] dW_o, db_o;
        public double[][] dW_y, db_y;

        // Memory variables for Adagrad
        public double[][] mW_f, mb_f;
        public double[][] mW_i, mb_i;
        public double[][] mW_C, mb_C;
        public double[][] mW_o, mb_o;
        public double[][] mW_y, mb_y;

        // Train previous variables
        public double[][] g_h_prev, g_C_prev;

        // LL parameters
        public List<double[][]> wparams0, dparams0, mparams0;

        public void Init()
        {
            // Initialize to a error of a random model
            smooth_loss = -Math.Log(1.0 / X_size) * T_steps; 

            // Init weights
            W_f = Ops.Randn(H_size, z_size, weight_sd, 0.5);
            b_f = Ops.Zeros(H_size, 1);

            W_i = Ops.Randn(H_size, z_size, weight_sd, 0.5);
            b_i = Ops.Zeros(H_size, 1);

            W_C = Ops.Randn(H_size, z_size, weight_sd, 0.0);
            b_C = Ops.Zeros(H_size, 1);

            W_o = Ops.Randn(H_size, z_size, weight_sd, 0.5);
            b_o = Ops.Zeros(H_size, 1);

            // For final layer to predict the next character
            W_y = Ops.Randn(X_size, H_size, weight_sd, 0.5);
            b_y = Ops.Zeros(X_size, 1);

            // Gradients
            dW_f = Ops.Zeros(W_f);
            dW_i = Ops.Zeros(W_i);
            dW_C = Ops.Zeros(W_C);
            dW_o = Ops.Zeros(W_o);
            dW_y = Ops.Zeros(W_y);

            db_f = Ops.Zeros(b_f);
            db_i = Ops.Zeros(b_i);
            db_C = Ops.Zeros(b_C);
            db_o = Ops.Zeros(b_o);
            db_y = Ops.Zeros(b_y);

            // Adagrad cache
            mW_f = Ops.Zeros(W_f);
            mW_i = Ops.Zeros(W_i);
            mW_C = Ops.Zeros(W_C);
            mW_o = Ops.Zeros(W_o);
            mW_y = Ops.Zeros(W_y);

            mb_f = Ops.Zeros(b_f);
            mb_i = Ops.Zeros(b_i);
            mb_C = Ops.Zeros(b_C);
            mb_o = Ops.Zeros(b_o);
            mb_y = Ops.Zeros(b_y);

            // Train previous variables
            g_h_prev = Ops.Zeros(H_size, 1);
            g_C_prev = Ops.Zeros(H_size, 1);


            wparams0 = new List<double[][]>() { W_f, W_i, W_C, W_o, W_y, b_f, b_i, b_C, b_o, b_y };
            dparams0 = new List<double[][]>() { dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y };
            mparams0 = new List<double[][]>() { mW_f, mW_i, mW_C, mW_o, mW_y, mb_f, mb_i, mb_C, mb_o, mb_y };
        }

        public (double[][], double[][], double[][], double[][], double[][], double[][], double[][], double[][], double[][]) Forward(double[][] x, double[][] h_prev, double[][] C_prev)
        {
            Ops.Assert(x, (X_size, 1)); // assert x.shape == (X_size, 1)
            Ops.Assert(h_prev, (H_size, 1)); // assert h_prev.shape == (H_size, 1)
            Ops.Assert(C_prev, (H_size, 1)); // assert C_prev.shape == (H_size, 1)

            var z = Ops.RowStack(h_prev, x); // z = np.row_stack((h_prev, x))
            var f = Ops.Sigmoid(Ops.Add(Ops.Dot(W_f, z), b_f)); // f = sigmoid(np.dot(W_f, z) + b_f)
            var i = Ops.Sigmoid(Ops.Add(Ops.Dot(W_i, z), b_i)); // i = sigmoid(np.dot(W_i, z) + b_i)
            var C_bar = Ops.Tanh(Ops.Add(Ops.Dot(W_C, z), b_C)); // C_bar = tanh(np.dot(W_C, z) + b_C)

            var C = Ops.Add(Ops.Mul(f, C_prev), Ops.Mul(i, C_bar)); // C = f * C_prev + i * C_bar
            var o = Ops.Sigmoid(Ops.Add(Ops.Dot(W_o, z), b_o)); // o = sigmoid(np.dot(W_o, z) + b_o)
            var h = Ops.Mul(o, Ops.Tanh(C)); // h = o * tanh(C)

            var y = Ops.Add(Ops.Dot(W_y, h), b_y); // y = np.dot(W_y, h) + b_y
            var p = Ops.Div(Ops.Exp(y), Ops.Sum(Ops.Exp(y))); // p = np.exp(y) / np.sum(np.exp(y))

            return (z, f, i, C_bar, C, o, h, y, p); // return z, f, i, C_bar, C, o, h, y, p
        }

        public (double[][], double[][]) Backward(int target, double[][] dh_next, double[][] dC_next, double[][] C_prev, double[][] z, double[][] f, double[][] i, double[][] C_bar, double[][] C, double[][] o, double[][] h, double[][] y, double[][] p)
        {
            Ops.Assert(z, (z_size, 1)); // assert z.shape == (X_size + H_size, 1)
            Ops.Assert(y, (X_size, 1)); // assert y.shape == (X_size, 1)
            Ops.Assert(p, (X_size, 1)); // assert p.shape == (X_size, 1)

            foreach (var param in new List<double[][]>() { dh_next, dC_next, C_prev, f, i, C_bar, C, o, h }) // for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
                Ops.Assert(param, (H_size, 1)); // assert param.shape == (H_size, 1)

            var dy = Ops.Copy(p); // dy = np.copy(p)
            dy[target][0] -= 1;   // dy[target] -= 1

            Ops.AddRef(dW_y, Ops.Dot(dy, Ops.T(h)));        // dW_y += np.dot(dy, h.T)
            Ops.AddRef(db_y, dy);                           // db_y += dy

            var dh = Ops.Dot(Ops.T(W_y), dy);               // dh = np.dot(W_y.T, dy)
            Ops.AddRef(dh, dh_next);                        // dh += dh_next
            var d_o = Ops.Mul(dh, Ops.Tanh(C));             // do = dh * tanh(C)
            d_o = Ops.Mul(Ops.DSigmoid(o), d_o);            // do = dsigmoid(o) * do
            Ops.AddRef(dW_o, Ops.Dot(d_o, Ops.T(z)));       // dW_o += np.dot(do, z.T)
            Ops.AddRef(db_o, d_o);                          // db_o += do

            var dC = Ops.Copy(dC_next);                     // dC = np.copy(dC_next)
            Ops.AddRef(
                    dC, Ops.Mul(dh, Ops.Mul(o, Ops.DTanh(Ops.Tanh(C))))
                );                                          // dC += dh * o * dtanh(tanh(C))
            var dC_bar = Ops.Mul(dC, i);                    // dC_bar = dC * i
            dC_bar = Ops.Mul(dC_bar, Ops.DTanh(C_bar));     // dC_bar = dC_bar * dtanh(C_bar)
            Ops.AddRef(dW_C, Ops.Dot(dC_bar, Ops.T(z)));    // dW_C += np.dot(dC_bar, z.T)
            Ops.AddRef(db_C, dC_bar);                       // db_C += dC_bar

            var di = Ops.Mul(dC, C_bar);                    // di = dC * C_bar
            di = Ops.Mul(Ops.DSigmoid(i), di);              // di = dsigmoid(i) * di
            Ops.AddRef(dW_i, Ops.Dot(di, Ops.T(z)));        // dW_i += np.dot(di, z.T)
            Ops.AddRef(db_i, di);                           // db_i += di

            var df = Ops.Mul(dC, C_prev);                   // df = dC * C_prev
            df = Ops.Mul(Ops.DSigmoid(f), df);              // df = dsigmoid(f) * df
            Ops.AddRef(dW_f, Ops.Dot(df, Ops.T(z)));        // dW_f += np.dot(df, z.T)
            Ops.AddRef(db_f, df);                           // db_f += df

            var dot1 = Ops.Dot(Ops.T(W_f), df);             // dz = np.dot(W_f.T, df) \
            var dot2 = Ops.Dot(Ops.T(W_i), di);             //     +np.dot(W_i.T, di) \
            var dot3 = Ops.Dot(Ops.T(W_C), dC_bar);         //     +np.dot(W_C.T, dC_bar) \
            var dot4 = Ops.Dot(Ops.T(W_o), d_o);            //     +np.dot(W_o.T, do)
            var dz = Ops.Add(dot1, Ops.Add(dot2, Ops.Add(dot3, dot4)));
            var dh_prev = Ops.Subarray(dz, H_size);         // dh_prev = dz[:H_size, :]
            var dC_prev = Ops.Mul(f, dC);                   // dC_prev = f * dC

            return (dh_prev, dC_prev);                      // return dh_prev, dC_prev
        }

        public (double, double[][], double[][]) ForwardBackward(int[] inputs, int[] targets, double[][] h_prev, double[][] C_prev)
        {
            // To store the values for each time step 
            // x_s, z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, y_s, p_s = { }, { }, { }, { }, { }, { }, { }, { }, { }, { }

            Dictionary<int, double[][]> x_s = new Dictionary<int, double[][]>(),
                z_s = new Dictionary<int, double[][]>(),
                f_s = new Dictionary<int, double[][]>(),
                i_s = new Dictionary<int, double[][]>(),
                C_bar_s = new Dictionary<int, double[][]>(),
                C_s = new Dictionary<int, double[][]>(),
                o_s = new Dictionary<int, double[][]>(),
                h_s = new Dictionary<int, double[][]>(),
                y_s = new Dictionary<int, double[][]>(),
                p_s = new Dictionary<int, double[][]>();

            // # Values at t - 1
            h_s[-1] = Ops.Copy(h_prev);      // h_s[-1] = np.copy(h_prev)
            C_s[-1] = Ops.Copy(C_prev);      // C_s[-1] = np.copy(C_prev)

            double loss = 0.0;              // loss = 0
            Ops.Assert(inputs, T_steps);    // assert len(inputs) == T_steps

            // Loop through time steps
            for (int t = 0; t < T_steps; ++t)
            {
                x_s[t] = Ops.Zeros(X_size, 1);         // x_s[t] = np.zeros((X_size, 1))
                x_s[t][inputs[t]][0] = 1.0;            // x_s[t][inputs[t]] = 1 # Input character

                // z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], y_s[t], p_s[t] \
                //      = forward(x_s[t], h_s[t - 1], C_s[t - 1]) # Forward pass
                (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], y_s[t], p_s[t]) = Forward(x_s[t], h_s[t - 1], C_s[t - 1]);

                loss += -Math.Log(p_s[t][targets[t]][0]); // loss += -np.log(p_s[t][targets[t], 0]) # Loss for at t
            }

            //Console.WriteLine($"loss ({loss})");

            // for dparam in [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y]:
            //      dparam.fill(0)
            foreach (var dparams in new List<double[][]>() { dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y })
                Ops.Fill(dparams, 0.0);

            var dh_next = Ops.Zeros(h_s[0]);    // dh_next = np.zeros_like(h_s[0]) #dh from the next character
            var dC_next = Ops.Zeros(C_s[0]);    // dC_next = np.zeros_like(C_s[0]) #dh from the next character

            /*
            for t in reversed(range(len(inputs))):
                # Backward pass
                dh_next, dC_next = backward(target = targets[t], dh_next = dh_next, dC_next = dC_next, C_prev = C_s[t-1],
                         z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t], C = C_s[t], o = o_s[t],
                         h = h_s[t], y = y_s[t], p = p_s[t])
             */
            for (int t = T_steps - 1; t >= 0; --t)
            {
                (dh_next, dC_next) = Backward(targets[t], dh_next, dC_next, C_s[t - 1],
                    z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t],
                    h_s[t], y_s[t], p_s[t]);
            }

            // Clip gradients to mitigate exploding gradients
            // for dparam in [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y]:
            foreach (var dparam in new List<double[][]>() { dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y })
                Ops.Clip(dparam, -1, 1); // np.clip(dparam, -1, 1, out= dparam)

            return (loss, h_s[T_steps - 1], C_s[T_steps - 1]); // return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]
        }

        public int[] Predict(int first_char_idx, int sentence_length)
        {
            var x = Ops.Zeros(X_size, 1);           // x = np.zeros((X_size, 1))
            x[first_char_idx][0] = 1.0;             // x[first_char_idx] = 1

            var h = Ops.Copy(g_h_prev);             // h = h_prev
            var C = Ops.Copy(g_C_prev);             // C = C_prev

            List<int> indexes = new List<int>();    // indexes = []
            indexes.Add(first_char_idx);
            double[][] p = null;

            for (int t = 0; t < sentence_length; ++t)
            {
                (_, _, _, _, C, _, h, _, p) = Forward(x, h, C);                             // _, _, _, _, C, _, h, _, p = forward(x, h, C)
                int idx = Ops.Choice(Ops.Range(X_size), p.SelectMany(a => a).ToArray());    // idx = np.random.choice(range(X_size), p=p.ravel())
                x = Ops.Zeros(X_size, 1);                                                   // x = np.zeros((X_size, 1))
                x[idx][0] = 1.0;                                                            // x[idx] = 1
                indexes.Add(idx);                                                           // indexes.append(idx)
            }

            return indexes.ToArray();
        }

        public void TrainOnBatch(int[] inputs, int[] targets, bool reset = false)
        {
            if (reset)
            {
                g_h_prev = Ops.Zeros(H_size, 1);
                g_C_prev = Ops.Zeros(H_size, 1);
            }

            double loss = 0.0;
            // loss, g_h_prev, g_C_prev = forward_backward(inputs, targets, g_h_prev, g_C_prev)
            // smooth_loss = smooth_loss * 0.999 + loss * 0.001
            (loss, g_h_prev, g_C_prev) = ForwardBackward(inputs, targets, g_h_prev, g_C_prev);
            //Console.WriteLine($"{smooth_loss:F6} {loss:F6}");
            smooth_loss = smooth_loss * 0.999 + loss * 0.001;

            /*
            # Update weights
            for param, dparam, mem in zip([W_f, W_i, W_C, W_o, W_y, b_f, b_i, b_C, b_o, b_y],
                                          [dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y],
                                          [mW_f, mW_i, mW_C, mW_o, mW_y, mb_f, mb_i, mb_C, mb_o, mb_y]):
                mem += dparam * dparam # Calculate sum of gradients
                #print(learning_rate * dparam)
                param += -(learning_rate * dparam / np.sqrt(mem + 1e-8))
             */

            var wparams = new List<double[][]>() {  W_f,  W_i,  W_C,  W_o,  W_y,  b_f,  b_i,  b_C,  b_o,  b_y };
            var dparams = new List<double[][]>() { dW_f, dW_i, dW_C, dW_o, dW_y, db_f, db_i, db_C, db_o, db_y };
            var mparams = new List<double[][]>() { mW_f, mW_i, mW_C, mW_o, mW_y, mb_f, mb_i, mb_C, mb_o, mb_y };
            for(int k = 0; k < 10; ++k)
            {
                Ops.AddRef(mparams[k], Ops.Mul(dparams[k], dparams[k]));
                Ops.AddRef(wparams[k], Ops.Mul(Ops.Div(dparams[k], Ops.Sqrt(mparams[k])), -learning_rate));
            }

        }
    }
}
