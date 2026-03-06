namespace CSharpAIAgentLab.Application.Abstractions;

public interface ILlmServiceResolver
{
    ILlmService Resolve(string providerName);
}
