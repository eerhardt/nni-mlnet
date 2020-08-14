// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections;
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

            Dictionary<string, string> parameters = GetDefaultParameters();
            Nni nni = new Nni();
            var receivedParameters = nni.GetNextParameter();
            parameters.Update(receivedParameters);

            var trainer = ctx.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options()
            {
                NumberOfIterations = int.Parse(parameters["NumberOfIterations"]),
                LearningRate = double.Parse(parameters["LearningRate"]),
                NumberOfLeaves = int.Parse(parameters["NumberOfLeaves"]),
                UseSoftmax = bool.Parse(parameters["UseSoftmax"]),
                Booster = new GradientBooster.Options()
                {
                    L2Regularization = double.Parse(parameters["L2Regularization"]),
                    L1Regularization = double.Parse(parameters["L1Regularization"]),
                }
            });

            var model = trainer.Fit(trainSet);
            var metrics = ctx.MulticlassClassification.Evaluate(model.Transform(testSet));
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
            Console.WriteLine($"TopKAccuracy: {metrics.TopKAccuracy}");

            nni.ReportFinalResult(metrics.MicroAccuracy);
        }

        private static Dictionary<string, string> GetDefaultParameters() =>
            new Dictionary<string, string>()
            {
                { "NumberOfIterations", "100" },
                { "LearningRate", ".25" },
                { "NumberOfLeaves", "30" },
                { "UseSoftmax", "false" },
                { "L2Regularization", "0.1" },
                { "L1Regularization", "0" },
            };

        private static void Update(this Dictionary<string, string> target, Dictionary<string, string> updates)
        {
            foreach(var kvp in updates)
            {
                target[kvp.Key] = kvp.Value;
            }
        }
    }
}
