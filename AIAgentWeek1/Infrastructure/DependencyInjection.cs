using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Application.Services;
using CSharpAIAgentLab.Infrastructure.LlmProviders;
using Microsoft.Extensions.DependencyInjection;

namespace CSharpAIAgentLab.Infrastructure;

public static class DependencyInjection
{
    public static IServiceCollection AddLlmInfrastructure(this IServiceCollection services)
    {
        services.AddSingleton<HttpClient>();
        services.AddTransient<OpenAiLlmService>();
        services.AddTransient<OllamaLlmService>();
        services.AddTransient<ILlmService>(sp => sp.GetRequiredService<OpenAiLlmService>());
        services.AddTransient<ILlmService>(sp => sp.GetRequiredService<OllamaLlmService>());
        services.AddTransient<ILlmServiceResolver, LlmServiceResolver>();
        services.AddTransient<RagDemoRunner>();

        return services;
    }
}
