using System.Net.Http;
using System.Net.Http.Json;

Console.Write("Ask your question: ");
var question = Console.ReadLine();

var client = new HttpClient();
var response = await client.GetAsync($"http://localhost:8000/ask?q={Uri.EscapeDataString(question!)}");
var result = await response.Content.ReadFromJsonAsync<AnswerResponse>();

Console.WriteLine("\nAnswer:");
Console.WriteLine(result?.Answer);

record AnswerResponse(string Answer);