using System.Text;
using System.Text.Json;
using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Application.Models;

namespace CSharpAIAgentLab.Infrastructure.LlmProviders;

public sealed class OllamaLlmService(HttpClient httpClient) : ILlmService
{
    public string ProviderName => "ollama";

    public async Task<LlmResult> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
    {
        var requestBody = new
        {
            model = "qwen2.5:1.5b",
            prompt,
            stream = false
        };

        var json = JsonSerializer.Serialize(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response;
        try
        {
            response = await httpClient.PostAsync("http://localhost:11434/api/generate", content, cancellationToken);
        }
        catch (HttpRequestException ex)
        {
            return new LlmResult(false, "로컬 서버 연결에 실패했습니다. Ollama 앱 실행 후 `ollama run qwen2.5:1.5b`를 먼저 시도해보세요.\n" + ex.Message);
        }

        var responseText = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return new LlmResult(false, $"요청 실패: {(int)response.StatusCode} {response.StatusCode}\n{responseText}");
        }

        using var doc = JsonDocument.Parse(responseText);
        if (doc.RootElement.TryGetProperty("response", out var outputTextElement))
        {
            return new LlmResult(true, outputTextElement.GetString() ?? string.Empty);
        }

        return new LlmResult(false, $"response 필드를 찾지 못했습니다. 원본 응답:\n{responseText}");
    }
}
