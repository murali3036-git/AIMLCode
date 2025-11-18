using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public class InsuranceData
    {
        [LoadColumn(0)]
        public float Age { get; set; }
        [LoadColumn(1)]
        public float Premium { get; set; }
    }
    public class InsurancePrediction
    {
        [ColumnName("Score")]
        public float PredictedPremium { get; set; }
    }
    public class FruitData
    {
        public float Weight { get; set; }
        public string Color { get; set; }   
        public bool IsApple { get; set; }   // True = Apple, False = Banana
        public string FruitType { get; set; }   

    }

    public class FruitPrediction
    {
        public string PredictedLabel { get; set; }
        //public bool IsApple { get; set; }
    }

    public class CustomerData
    {
        public float Age { get; set; }
        public float Spending { get; set; }
    }

    public class CustomerCluster
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }
        //public float[] Score { get; set; }
    }
    public class NiftyLagData
    {
        [LoadColumn(0)] public string Date { get; set; }
        [LoadColumn(1)] public float Nifty { get; set; }

        [LoadColumn(2)] public float NiftyLag1 { get; set; }
        [LoadColumn(3)] public float NiftyLag2 { get; set; }
        [LoadColumn(4)] public float NiftyLag3 { get; set; }
    }

    public class NiftyPrediction
    {
        [ColumnName("Score")]
        public float PredictedValue { get; set; }
    }
}
