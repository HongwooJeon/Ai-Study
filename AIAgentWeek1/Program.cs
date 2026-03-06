using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Infrastructure;
using Microsoft.Extensions.DependencyInjection;

var providerName = (args.FirstOrDefault() ?? Environment.GetEnvironmentVariable("LLM_PROVIDER") ?? "ollama")
    .Trim()
    .ToLowerInvariant();

var prompt = "C#에서 LLM API를 호출하는 간단한 예제를 설명해줘.";

var services = new ServiceCollection();
services.AddLlmInfrastructure();

using var serviceProvider = services.BuildServiceProvider();
var resolver = serviceProvider.GetRequiredService<ILlmServiceResolver>();

ILlmService llmService;
try
{
    llmService = resolver.Resolve(providerName);
}
catch (ArgumentException ex)
{
    Console.WriteLine(ex.Message);
    return;
}

var result = await llmService.GenerateAsync(prompt);
if (result.IsSuccess)
{
    Console.WriteLine($"[{llmService.ProviderName}] 응답:");
    Console.WriteLine(result.Message);
    return;
}

Console.WriteLine($"[{llmService.ProviderName}] 요청 실패:");
Console.WriteLine(result.Message);


