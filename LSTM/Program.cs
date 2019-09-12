using System;
using System.Diagnostics;
using System.Linq;

namespace LSTMsharp
{
    class MainClass
    {

        static void RunLSTM()
        {
            int nbWord = 3;
            string word = "Hello World\n";
            string word0 = Enumerable.Repeat(word, nbWord).Glue("");

            var udata = word0.Distinct().ToArray();
            var tableChar2Idx = udata.Select((c, i) => (c, i)).ToDictionary(a => a.c, b => b.i);
            var tableIdx2Char = tableChar2Idx.ToDictionary(a => a.Value, b => b.Key);

            int iteration = 0;
            int epochs = 4000;
            int displayEpoch = 250;
            int p = 0;

            int X_size = udata.Length;
            int H_size = X_size * 2;
            int T_steps = word.Length;
            Console.WriteLine($"X_size:{X_size} H_size:{H_size} T_steps:{T_steps}");

            var lstm = new LSTM(X_size, H_size, T_steps);
            var sw = Stopwatch.StartNew();
            while (iteration <= epochs)
            {
                if (p + T_steps >= word0.Length) p = 0;

                var inputs = Enumerable.Range(p, T_steps).Select(i => tableChar2Idx[word0[i]]).ToArray();
                var targets = Enumerable.Range(p + 1, T_steps).Select(i => tableChar2Idx[word0[i]]).ToArray();

                lstm.TrainOnBatch(inputs, targets, p == 0);

                if (iteration % displayEpoch == 0)
                {
                    Console.WriteLine($"Epochs:{iteration,6}/{epochs} Loss:{lstm.smooth_loss:F6} Time:{sw.ElapsedMilliseconds,6} ms");
                    var r = lstm.Predict(inputs[0], T_steps * 3 - 2);
                    Console.WriteLine(r.Select(i => tableIdx2Char[i]).Glue(""));
                }

                ++iteration;
                p += T_steps;
            }
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            RunLSTM();
        }
    }
}
