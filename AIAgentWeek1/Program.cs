using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Application.Services;
using CSharpAIAgentLab.Infrastructure;
using Microsoft.Extensions.DependencyInjection;

var isRagMode = string.Equals(args.FirstOrDefault(), "rag", StringComparison.OrdinalIgnoreCase);

var services = new ServiceCollection();
services.AddLlmInfrastructure();

using var serviceProvider = services.BuildServiceProvider();

if (isRagMode)
{
    var query = args.Skip(1).FirstOrDefault()?.Trim();
    if (string.IsNullOrWhiteSpace(query))
    {
        query = "vector db를 활용해서 프롬프트를 만드는 핵심 방법을 알려줘";
    }

    // Optional third argument for chat model in rag mode, e.g. qwen2.5:3b
    var ragChatModelArg = args.Skip(2).FirstOrDefault();
    if (!string.IsNullOrWhiteSpace(ragChatModelArg))
    {
        Environment.SetEnvironmentVariable("OLLAMA_MODEL", ragChatModelArg.Trim());
    }

    var runner = serviceProvider.GetRequiredService<RagDemoRunner>();
    var ragResult = await runner.RunAsync(query);

    if (ragResult.IsSuccess)
    {
        Console.WriteLine("[rag] 응답:");
        Console.WriteLine(ragResult.Message);
        return;
    }

    Console.WriteLine("[rag] 요청 실패:");
    Console.WriteLine(ragResult.Message);
    return;
}

var providerName = (args.FirstOrDefault() ?? Environment.GetEnvironmentVariable("LLM_PROVIDER") ?? "ollama")
    .Trim()
    .ToLowerInvariant();

// Optional second argument for model name, e.g. qwen2.5:3b
var modelNameArg = args.Skip(1).FirstOrDefault();
if (!string.IsNullOrWhiteSpace(modelNameArg))
{
    Environment.SetEnvironmentVariable("OLLAMA_MODEL", modelNameArg.Trim());
}

var prompt = "c# 으로 AI 에이전트 공부 순서 알려줘";
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


