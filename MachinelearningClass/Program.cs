using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
namespace MachinelearningClass
{
    internal class Program
    {
        public static string datapath = "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\";
         static void Main(string[] args)
        {
            Week4.Lab18_RAGChatGPTOnline().Wait();
           //Week4.Lab18_RAGChatGPTOnline().Wait();
            //Week4.MockInterview().Wait();
            //Week4.Lab15_BertEncoding();
            //Week4.Lab17_ChatGPTOnline().Wait();
            Console.ReadLine();
        }
    }
    
}
