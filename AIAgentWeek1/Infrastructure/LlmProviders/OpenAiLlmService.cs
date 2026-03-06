using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Application.Models;

namespace CSharpAIAgentLab.Infrastructure.LlmProviders;

public sealed class OpenAiLlmService(HttpClient httpClient) : ILlmService
{
    public string ProviderName => "openai";

    public async Task<LlmResult> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return new LlmResult(false, "OPENAI_API_KEY 환경 변수를 먼저 설정해주세요.");
        }

        httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        var requestBody = new
        {
            model = "gpt-4o-mini",
            input = prompt
        };

        var json = JsonSerializer.Serialize(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response;
        try
        {
            response = await httpClient.PostAsync("https://api.openai.com/v1/responses", content, cancellationToken);
        }
        catch (HttpRequestException ex)
        {
            return new LlmResult(false, $"네트워크 오류로 요청하지 못했습니다. {ex.Message}");
        }

        var responseText = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return new LlmResult(false, $"요청 실패: {(int)response.StatusCode} {response.StatusCode}\n{responseText}");
        }

        using var doc = JsonDocument.Parse(responseText);
        if (doc.RootElement.TryGetProperty("output_text", out var outputTextElement))
        {
            return new LlmResult(true, outputTextElement.GetString() ?? string.Empty);
        }

        return new LlmResult(false, $"output_text를 찾지 못했습니다. 원본 응답:\n{responseText}");
    }
}
