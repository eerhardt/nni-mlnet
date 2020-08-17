// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using Microsoft.ML.AutoML;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.Nni
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            MLContext ctx = new MLContext();
            var columnInference = ctx.Auto().InferColumns("digits.csv", labelColumnIndex: 64, separatorChar: ',');

            IDataView data = ctx.Data.LoadFromTextFile("digits.csv", columnInference.TextLoaderOptions);
            var trainTestSplit = ctx.Data.TrainTestSplit(data, testFraction: .25);

            var preprocessPipeline = ctx.Transforms.NormalizeMeanVariance("Features")
                .Append(ctx.Transforms.Conversion.MapValueToKey("Label"))
                .Fit(trainTestSplit.TrainSet);
            IDataView trainSet = preprocessPipeline.Transform(trainTestSplit.TrainSet);
            IDataView testSet = preprocessPipeline.Transform(trainTestSplit.TestSet);

            Nni nni = new Nni();
            Dictionary<string, string> parameters = nni.GetNextParameter();

            var trainer = ctx.MulticlassClassification.Trainers.LightGbm(CreateOptions(parameters));

            var model = trainer.Fit(trainSet);
            var metrics = ctx.MulticlassClassification.Evaluate(model.Transform(testSet));
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
            Console.WriteLine($"TopKAccuracy: {metrics.TopKAccuracy}");

            nni.ReportFinalResult(metrics.MicroAccuracy);
        }

        private static LightGbmMulticlassTrainer.Options CreateOptions(Dictionary<string, string> parameters)
        {
            var options = new LightGbmMulticlassTrainer.Options();
            if (parameters.TryGetValue("NumberOfIterations", out string numberOfIterations))
            {
                options.NumberOfIterations = int.Parse(numberOfIterations);
            }
            if (parameters.TryGetValue("LearningRate", out string learningRate))
            {
                options.LearningRate = double.Parse(learningRate);
            }
            if (parameters.TryGetValue("NumberOfLeaves", out string numberOfLeaves))
            {
                options.NumberOfLeaves = ParseInt(numberOfLeaves);
            }
            if (parameters.TryGetValue("UseSoftmax", out string useSoftmax))
            {
                options.UseSoftmax = bool.Parse(useSoftmax);
            }

            options.Booster = new GradientBooster.Options();
            if (parameters.TryGetValue("L2Regularization", out string l2Regularization))
            {
                options.Booster.L2Regularization = double.Parse(l2Regularization);
            }
            if (parameters.TryGetValue("L1Regularization", out string l1Regularization))
            {
                options.Booster.L1Regularization = double.Parse(l1Regularization);
            }

            return options;
        }

        private static int ParseInt(string value)
        {
            if (int.TryParse(value, out int intValue))
            {
                return intValue;
            }
            else if (float.TryParse(value, out float floatValue))
            {
                return (int)floatValue;
            }

            throw new FormatException($"Can't parse '{value}' into an int.");
        }
    }
}
