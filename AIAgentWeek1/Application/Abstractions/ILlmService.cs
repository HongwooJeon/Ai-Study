using CSharpAIAgentLab.Application.Models;

namespace CSharpAIAgentLab.Application.Abstractions;

public interface ILlmService
{
    string ProviderName { get; }

    Task<LlmResult> GenerateAsync(string prompt, CancellationToken cancellationToken = default);
}
