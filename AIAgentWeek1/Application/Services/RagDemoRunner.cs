using System.Text;
using System.Text.Json;
using CSharpAIAgentLab.Application.Abstractions;
using CSharpAIAgentLab.Application.Models;

namespace CSharpAIAgentLab.Application.Services;

public sealed class RagDemoRunner(HttpClient httpClient, ILlmServiceResolver llmServiceResolver)
{
    public async Task<LlmResult> RunAsync(string query, CancellationToken cancellationToken = default)
    {
        var qdrantUrl = Environment.GetEnvironmentVariable("QDRANT_URL")?.Trim() ?? "http://localhost:6333";
        var collectionName = Environment.GetEnvironmentVariable("QDRANT_COLLECTION")?.Trim() ?? "study_docs";
        var embeddingModel = Environment.GetEnvironmentVariable("OLLAMA_EMBED_MODEL")?.Trim() ?? "nomic-embed-text";
        var vectorDbMode = Environment.GetEnvironmentVariable("VECTOR_DB_MODE")?.Trim().ToLowerInvariant() ?? "qdrant";

        var docs = LoadMarkdownChunks();
        if (docs.Length == 0)
        {
            docs = GetLearningDocs();
        }

        var docEmbeddings = new List<float[]>();

        foreach (var doc in docs)
        {
            var embeddingResult = await CreateEmbeddingAsync(embeddingModel, doc, cancellationToken);
            if (!embeddingResult.IsSuccess)
            {
                return new LlmResult(false, embeddingResult.Message);
            }

            docEmbeddings.Add(embeddingResult.Vector!);
        }

        var queryEmbeddingResult = await CreateEmbeddingAsync(embeddingModel, query, cancellationToken);
        if (!queryEmbeddingResult.IsSuccess)
        {
            return new LlmResult(false, queryEmbeddingResult.Message);
        }

        List<RagSearchHit> hits;
        if (vectorDbMode == "memory")
        {
            hits = SearchInMemory(docs, docEmbeddings, queryEmbeddingResult.Vector!, 3);
        }
        else
        {
            var vectorSize = docEmbeddings[0].Length;
            var collectionResult = await EnsureCollectionAsync(qdrantUrl, collectionName, vectorSize, cancellationToken);
            if (!collectionResult.IsSuccess)
            {
                return new LlmResult(false,
                    $"{collectionResult.Message}\nDocker가 없다면 `$env:VECTOR_DB_MODE=memory`로 메모리 모드를 사용하세요.");
            }

            var upsertResult = await UpsertDocumentsAsync(qdrantUrl, collectionName, docs, docEmbeddings, cancellationToken);
            if (!upsertResult.IsSuccess)
            {
                return new LlmResult(false,
                    $"{upsertResult.Message}\nDocker가 없다면 `$env:VECTOR_DB_MODE=memory`로 메모리 모드를 사용하세요.");
            }

            var searchResult = await SearchAsync(qdrantUrl, collectionName, queryEmbeddingResult.Vector!, cancellationToken);
            if (!searchResult.IsSuccess)
            {
                return new LlmResult(false,
                    $"{searchResult.Message}\nDocker가 없다면 `$env:VECTOR_DB_MODE=memory`로 메모리 모드를 사용하세요.");
            }

            hits = searchResult.Hits!;
        }

        var context = string.Join("\n\n", hits.Select((hit, index) =>
            $"[문서 {index + 1} | score={hit.Score:F4}]\n{hit.Text}"));

        var ragPrompt = $"""
다음은 Vector DB에서 검색된 참고 문서입니다.
아래 문서의 내용만 근거로 답변하고, 부족한 정보는 부족하다고 말해주세요.

[검색 문서]
{context}

[질문]
{query}

[요청]
- 한국어로 답변
- 실행 가능한 학습 순서 5단계
- 각 단계마다 실습 1개 포함
""";

        var llmService = llmServiceResolver.Resolve("ollama");
        return await llmService.GenerateAsync(ragPrompt, cancellationToken);
    }

    private static List<RagSearchHit> SearchInMemory(
        IReadOnlyList<string> docs,
        IReadOnlyList<float[]> vectors,
        float[] queryVector,
        int topK)
    {
        return docs
            .Select((doc, index) => new RagSearchHit(CosineSimilarity(queryVector, vectors[index]), doc))
            .OrderByDescending(hit => hit.Score)
            .Take(topK)
            .ToList();
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        var len = Math.Min(a.Length, b.Length);
        if (len == 0)
        {
            return 0;
        }

        double dot = 0;
        double normA = 0;
        double normB = 0;

        for (var i = 0; i < len; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0)
        {
            return 0;
        }

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    private static string[] GetLearningDocs()
    {
        return
        [
            "RAG의 핵심은 질문과 관련된 문서를 벡터 검색으로 찾아 프롬프트에 함께 넣는 것이다. 그래서 환각을 줄이고 근거 기반 답변을 만들 수 있다.",
            "벡터 DB를 사용할 때는 문서를 chunk로 나누고, chunk마다 embedding을 생성해 저장한다. 검색 시에는 질문 embedding으로 top-k chunk를 가져온다.",
            "좋은 chunk 전략은 300~800 토큰 범위와 10~20% overlap이다. 너무 짧으면 문맥이 끊기고, 너무 길면 검색 정밀도가 떨어진다.",
            "RAG 품질을 높이려면 재순위화(reranking), 메타데이터 필터, 하이브리드 검색(BM25+Vector)을 고려한다.",
            "C#에서는 먼저 간단한 파이프라인(문서 적재 -> 임베딩 -> upsert -> 검색 -> 프롬프트 구성)을 만들고, 이후 배치 임베딩과 캐싱을 붙이는 것이 좋다."
        ];
    }

    private static string[] LoadMarkdownChunks()
    {
        var docsDir = Environment.GetEnvironmentVariable("RAG_DOCS_DIR")?.Trim();
        if (string.IsNullOrWhiteSpace(docsDir) || !Directory.Exists(docsDir))
        {
            return [];
        }

        var chunkSize = int.TryParse(Environment.GetEnvironmentVariable("RAG_CHUNK_SIZE"), out var parsedChunkSize)
            ? Math.Max(300, parsedChunkSize)
            : 900;

        var overlap = int.TryParse(Environment.GetEnvironmentVariable("RAG_CHUNK_OVERLAP"), out var parsedOverlap)
            ? Math.Max(0, parsedOverlap)
            : 120;

        if (overlap >= chunkSize)
        {
            overlap = chunkSize / 5;
        }

        var files = Directory.GetFiles(docsDir, "*.md", SearchOption.AllDirectories);
        var chunks = new List<string>();

        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            foreach (var chunk in ChunkText(text, chunkSize, overlap))
            {
                chunks.Add($"[source:{Path.GetFileName(file)}]\n{chunk}");
            }
        }

        return chunks.ToArray();
    }

    private static IEnumerable<string> ChunkText(string text, int chunkSize, int overlap)
    {
        var normalized = text.Replace("\r\n", "\n").Trim();
        if (string.IsNullOrWhiteSpace(normalized))
        {
            yield break;
        }

        var start = 0;
        while (start < normalized.Length)
        {
            var plannedEnd = Math.Min(start + chunkSize, normalized.Length);
            var end = plannedEnd;

            if (plannedEnd < normalized.Length)
            {
                var windowLength = plannedEnd - start;
                var breakPos = normalized.LastIndexOf('\n', plannedEnd - 1, windowLength);
                if (breakPos > start + (chunkSize / 2))
                {
                    end = breakPos + 1;
                }
            }

            var chunk = normalized[start..end].Trim();
            if (!string.IsNullOrWhiteSpace(chunk))
            {
                yield return chunk;
            }

            if (end >= normalized.Length)
            {
                yield break;
            }

            start = Math.Max(end - overlap, start + 1);
        }
    }

    private async Task<(bool IsSuccess, float[]? Vector, string Message)> CreateEmbeddingAsync(
        string model,
        string text,
        CancellationToken cancellationToken)
    {
        var requestBody = new
        {
            model,
            prompt = text
        };

        var responseTextResult = await PostJsonAsync(
            "http://localhost:11434/api/embeddings",
            requestBody,
            cancellationToken,
            "Ollama 임베딩 요청 실패");

        if (!responseTextResult.IsSuccess)
        {
            return (false, null, responseTextResult.Message);
        }

        using var doc = JsonDocument.Parse(responseTextResult.Message);
        if (!doc.RootElement.TryGetProperty("embedding", out var embeddingElement) ||
            embeddingElement.ValueKind != JsonValueKind.Array)
        {
            return (false, null, "Ollama 응답에서 embedding 필드를 찾지 못했습니다.");
        }

        var values = new List<float>();
        foreach (var item in embeddingElement.EnumerateArray())
        {
            if (item.TryGetSingle(out var number))
            {
                values.Add(number);
                continue;
            }

            if (item.TryGetDouble(out var numberAsDouble))
            {
                values.Add((float)numberAsDouble);
                continue;
            }

            return (false, null, "embedding 값 파싱에 실패했습니다.");
        }

        return (true, values.ToArray(), string.Empty);
    }

    private async Task<LlmResult> EnsureCollectionAsync(
        string qdrantUrl,
        string collectionName,
        int vectorSize,
        CancellationToken cancellationToken)
    {
        var requestBody = new
        {
            vectors = new
            {
                size = vectorSize,
                distance = "Cosine"
            }
        };

        var response = await PutJsonAsync(
            $"{qdrantUrl}/collections/{collectionName}",
            requestBody,
            cancellationToken,
            "Qdrant 컬렉션 생성/확인 실패");

        return response;
    }

    private async Task<LlmResult> UpsertDocumentsAsync(
        string qdrantUrl,
        string collectionName,
        IReadOnlyList<string> docs,
        IReadOnlyList<float[]> vectors,
        CancellationToken cancellationToken)
    {
        var points = docs.Select((doc, index) => new
        {
            id = index + 1,
            vector = vectors[index],
            payload = new
            {
                text = doc,
                source = "rag-demo"
            }
        });

        var requestBody = new { points };

        return await PutJsonAsync(
            $"{qdrantUrl}/collections/{collectionName}/points?wait=true",
            requestBody,
            cancellationToken,
            "Qdrant 문서 upsert 실패");
    }

    private async Task<(bool IsSuccess, List<RagSearchHit>? Hits, string Message)> SearchAsync(
        string qdrantUrl,
        string collectionName,
        float[] queryVector,
        CancellationToken cancellationToken)
    {
        var requestBody = new
        {
            vector = queryVector,
            limit = 3,
            with_payload = true
        };

        var responseTextResult = await PostJsonAsync(
            $"{qdrantUrl}/collections/{collectionName}/points/search",
            requestBody,
            cancellationToken,
            "Qdrant 검색 실패");

        if (!responseTextResult.IsSuccess)
        {
            return (false, null, responseTextResult.Message);
        }

        using var doc = JsonDocument.Parse(responseTextResult.Message);
        if (!doc.RootElement.TryGetProperty("result", out var resultElement) || resultElement.ValueKind != JsonValueKind.Array)
        {
            return (false, null, "Qdrant 응답에서 result 배열을 찾지 못했습니다.");
        }

        var hits = new List<RagSearchHit>();
        foreach (var item in resultElement.EnumerateArray())
        {
            var score = item.TryGetProperty("score", out var scoreElement) && scoreElement.TryGetDouble(out var scoreValue)
                ? scoreValue
                : 0;

            var text = item.TryGetProperty("payload", out var payloadElement) &&
                       payloadElement.ValueKind == JsonValueKind.Object &&
                       payloadElement.TryGetProperty("text", out var textElement)
                ? textElement.GetString() ?? string.Empty
                : string.Empty;

            if (!string.IsNullOrWhiteSpace(text))
            {
                hits.Add(new RagSearchHit(score, text));
            }
        }

        if (hits.Count == 0)
        {
            return (false, null, "검색 결과가 비어 있습니다. 컬렉션/임베딩 모델 설정을 확인해주세요.");
        }

        return (true, hits, string.Empty);
    }

    private async Task<LlmResult> PutJsonAsync(
        string url,
        object requestBody,
        CancellationToken cancellationToken,
        string failPrefix)
    {
        var json = JsonSerializer.Serialize(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response;
        try
        {
            response = await httpClient.PutAsync(url, content, cancellationToken);
        }
        catch (HttpRequestException ex)
        {
            return new LlmResult(false, $"{failPrefix}: {ex.Message}");
        }

        var responseText = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return new LlmResult(false, $"{failPrefix}: {(int)response.StatusCode} {response.StatusCode}\n{responseText}");
        }

        return new LlmResult(true, responseText);
    }

    private async Task<(bool IsSuccess, string Message)> PostJsonAsync(
        string url,
        object requestBody,
        CancellationToken cancellationToken,
        string failPrefix)
    {
        var json = JsonSerializer.Serialize(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        HttpResponseMessage response;
        try
        {
            response = await httpClient.PostAsync(url, content, cancellationToken);
        }
        catch (HttpRequestException ex)
        {
            return (false, $"{failPrefix}: {ex.Message}");
        }

        var responseText = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return (false, $"{failPrefix}: {(int)response.StatusCode} {response.StatusCode}\n{responseText}");
        }

        return (true, responseText);
    }
}
