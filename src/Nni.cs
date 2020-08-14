// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Microsoft.ML.Nni
{
    internal class Nni
    {
        private readonly string _sysDir;
        private readonly string _trialJobId;
        private readonly string _nniPlatform;
        private int _parameterId;

        public Nni()
        {
            _sysDir = Environment.GetEnvironmentVariable("NNI_SYS_DIR");
            _trialJobId = Environment.GetEnvironmentVariable("NNI_TRIAL_JOB_ID");
            _nniPlatform = Environment.GetEnvironmentVariable("NNI_PLATFORM");
        }

        public Dictionary<string, string> GetNextParameter()
        {
            if (string.IsNullOrEmpty(_sysDir))
            {
                return new Dictionary<string, string>();
            }

            string paramsFilePath = Path.Combine(_sysDir, "parameter.cfg");
            using FileStream paramsFile = File.OpenRead(paramsFilePath);

            string fileContents = File.ReadAllText(paramsFilePath);
            ParameterConfig config = JsonSerializer.Deserialize<ParameterConfig>(fileContents);

            _parameterId = config.parameter_id;
            return config.parameters.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.ToString());
        }

        public void ReportFinalResult(double metric)
        {
            Result r = new Result()
            {
                parameter_id = _parameterId,
                trial_job_id = _trialJobId ?? "local",
                type = "FINAL",
                sequence = 0,
                value = metric.ToString(),
            };

            string data = JsonSerializer.Serialize(r);

            if (_nniPlatform != "local")
            {
                Console.WriteLine($"NNISDK_MEb'{data}'");
            }
            else
            {
                string sysDir = _sysDir ?? Environment.CurrentDirectory;
                string metricsDirectory = Path.Combine(sysDir, ".nni");
                string metricsFilePath = Path.Combine(metricsDirectory, "metrics");
                Directory.CreateDirectory(metricsDirectory);

                using FileStream metricsFile = new FileStream(metricsFilePath, FileMode.OpenOrCreate);
                using TextWriter writer = new StreamWriter(metricsFile);

                string linePrefix = $"ME{data.Length:D6}";
                writer.Write(linePrefix);
                writer.WriteLine(data);
                writer.Flush();
            }
        }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
        private class ParameterConfig
        {
            public int parameter_id { get; set; }
            public string parameter_source { get; set; }
            public Dictionary<string, object> parameters { get; set; }
            public int parameter_index { get; set; }
        }
        private class Result
        {
            public int parameter_id { get; set; }
            public string trial_job_id { get; set; }
            public string type { get; set; }
            public int sequence { get; set; }
            public string value { get; set; }
        }
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    }
}
