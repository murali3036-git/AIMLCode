using MachinelearningClass.InterviewQuestions;
using MachinelearningClass.ModelNLP;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using InputText = MachinelearningClass.InterviewQuestions.InputText;
using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.ML.Data;
using AllMiniLmL6V2Sharp;
using System.Data;
using OpenAI;
using OpenAI.Chat;
using Google.Protobuf;
using OpenAI.Embeddings;
using System.ClientModel;
using Newtonsoft.Json.Serialization;
using MachinelearningClass.DataNLP;
namespace MachinelearningClass
{
    public class Week4 // NLP
    {
        //The vocab.txt is used by the tokenizer to convert text into tokens (numbers),
        //and the ONNX model all-MiniLM-L6-v2.onnx is used by the embedder to
        //convert those tokens into vector embeddings that capture meaning.
        // https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/tree/main
        // Data folder does not have the All mini & GPT model and vocab json please download
        // from huggingface
        
        public static void Lab15_SimpleBertEncoding()
        {
            var tokenizer = new BertTokenizer(Program.datapath + @"\\vocab.txt");
            var embedder = new AllMiniLmL6V2Embedder(

                Program.datapath + @"\\Model.onnx",
                tokenizer
            );
            string texttobeMatched = "I love cricket and especially batting.";
            var texttobeMatchedV = embedder.GenerateEmbedding(texttobeMatched).ToArray();

            Console.WriteLine("Enter text be matched");
            string inputText = Console.ReadLine();
            var inputTextV = embedder.GenerateEmbedding(inputText).ToArray();
            Console.WriteLine(Common.CalculateCosineSimilarity(texttobeMatchedV, inputTextV));


        }
        public static void Lab16_FailedGPTEncoding()
        {
            var ml = new MLContext();
            string modelPath = Program.datapath + @"\\GPT\\model.onnx";

            // Create ONNX pipeline
            var pipeline = ml.Transforms.ApplyOnnxModel(
                outputColumnNames: new[] { "logits" },
                inputColumnNames: new[] { "input_ids" },
                modelFile: modelPath
            );

            // Create dummy input for Fit
            var dummyInput = new GPT2Input
            {
                input_ids = new long[1, 16] // all zeros
            };

            var model = pipeline.Fit(ml.Data.LoadFromEnumerable(new List<GPT2Input> { dummyInput }));

            var engine = ml.Model.CreatePredictionEngine<GPT2Input, GPT2Output>(model);

            // Example input: "I love" -> token IDs padded to length 16
            long[] tokens = { 40, 18435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var input = new GPT2Input
            {
                input_ids = new long[1, 16]
            };
            for (int i = 0; i < tokens.Length; i++)
                input.input_ids[0, i] = tokens[i];

            var result = engine.Predict(input);

            // Find the next token with the highest probability for last position
            int seqLen = result.logits.GetLength(1);
            int vocabSize = result.logits.GetLength(2);

            float max = float.MinValue;
            int predictedIndex = -1;

            for (int v = 0; v < vocabSize; v++)
            {
                float val = result.logits[0, seqLen - 1, v]; // last token position
                if (val > max)
                {
                    max = val;
                    predictedIndex = v;
                }
            }

            Console.WriteLine($"Predicted next token ID: {predictedIndex}");
        }

        public static async Task Lab17_SimpleChatGPTOnline()
        {
            var credential = new ApiKeyCredential(Environment.GetEnvironmentVariable("aikey"));
            var chatClient = new ChatClient("gpt-3.5-turbo", credential); // Use a standard model for generation

            string simpleSentencePrompt = "I love ";

            ChatCompletion completion = await chatClient.CompleteChatAsync(
                                        messages: new[]
                                        {
                                        new UserChatMessage(simpleSentencePrompt)
                                        }
                                        );

            string prediction = completion.Content.Last().Text;
            Console.WriteLine( prediction );
        }
        public static async Task Lab18_RAGChatGPTOnline()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);

            List<RAGLookup> lookupStore = DataforNlp.getRAGData();

            foreach (var item in lookupStore)
            {
                var embed = await embeddingClient.GenerateEmbeddingAsync(item.Description);
                item.DescriptionEmbedding = embed.Value.ToFloats().ToArray();

            }
            Console.WriteLine("Enter who you are ?");

            string candidateExp = Console.ReadLine();
            var candidateEmbed = await embeddingClient.GenerateEmbeddingAsync(candidateExp);
            var candidateVector = candidateEmbed.Value.ToFloats().ToArray();

            var bestMatch = lookupStore
                .OrderByDescending(x => Common.CalculateCosineSimilarity(x.DescriptionEmbedding, candidateVector))
                .First();



            // Start interview with selected questions
            var messages = new List<ChatMessage>();
           
           
            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new SystemChatMessage(
                $"Ask only: {bestMatch.QuestionstobeAsked}. " +
                "Display atleast 10 questions"));
            //messages.Add(new UserChatMessage(candidateExp));


            var completion = await chat.CompleteChatAsync(messages);
            string questionfromchatgpt = completion.Value.Content.Last().Text;
            Console.WriteLine($"{questionfromchatgpt}");
           

           


        }
        public static async Task Lab19_PromptUnderstanding()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            //var client = new OpenAIClient(key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            var messages = new List<ChatMessage>
            {
                new SystemChatMessage("Take c# interview . Only ask ASP.NET core question if he does not answr that ask him basic OOP. Do not repeat question once asked. Ask one question at a time. Do not answer yourself.")
            };
            while (true)
            {
                var completion = await chat.CompleteChatAsync(messages);
                string questionfromchatgpt = completion.Value.Content.Last().Text;
                messages.Add(new AssistantChatMessage(questionfromchatgpt));
                Console.WriteLine($"{questionfromchatgpt}");
                var userResponse = "";
                userResponse = Console.ReadLine(); // answr
                messages.Add(new UserChatMessage(userResponse)); // chat gpt

            }

            
        }

       


        public class BertInput
        {
            [VectorType]
            public long[] input_ids { get; set; }
            [VectorType]
            public long[] attention_mask { get; set; }
        }

        public class BertOutput
        {
            [VectorType]
            public float[] sentence_embedding { get; set; }
        }

        public class QAItem
        {
            public string Question { get; set; }
            public string Answer { get; set; }
            public float[] Embedding { get; set; }
        }


    }
}
