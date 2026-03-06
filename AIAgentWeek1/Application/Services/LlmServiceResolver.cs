using CSharpAIAgentLab.Application.Abstractions;

namespace CSharpAIAgentLab.Application.Services;

public sealed class LlmServiceResolver(IEnumerable<ILlmService> services) : ILlmServiceResolver
{
    private readonly IReadOnlyDictionary<string, ILlmService> _services =
        services.ToDictionary(service => service.ProviderName, StringComparer.OrdinalIgnoreCase);

    public ILlmService Resolve(string providerName)
    {
        if (_services.TryGetValue(providerName, out var service))
        {
            return service;
        }

        var availableProviders = string.Join(", ", _services.Keys.OrderBy(x => x));
        throw new ArgumentException(
            $"지원하지 않는 provider: {providerName}. 사용 가능 값: {availableProviders}",
            nameof(providerName));
    }
}
