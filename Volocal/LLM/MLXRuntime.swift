import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Loads an on-disk MLX pack and streams a chat reply.
/// The weights are a Hugging Face MLX folder, not a GGUF.
enum MLXRuntime {
    /// MLX keeps every temporary Metal buffer unless this is capped. On iPhone
    /// that grows for a few turns and then the process is killed.
    private static let configureCache: Void = {
        Memory.cacheLimit = 32 * 1024 * 1024
    }()

    static func load(directory: URL) async throws -> ModelContainer {
        _ = configureCache
        guard LLMModelSpec.mlxPackIsComplete(at: directory) else {
            throw MLXRuntimeError.incompletePack
        }
        return try await LLMModelFactory.shared.loadContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )
    }

    static func releaseTemporaryBuffers() {
        Memory.clearCache()
    }

    static func openStream(
        container: ModelContainer,
        system: String,
        history: [(role: String, content: String)]
    ) async throws -> AsyncStream<Generation> {
        var messages: [Chat.Message] = []
        let instructions = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !instructions.isEmpty {
            messages.append(.system(instructions))
        }
        for turn in history {
            let text = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            if turn.role == "assistant" {
                messages.append(.assistant(text))
            } else {
                messages.append(.user(text))
            }
        }
        guard !messages.isEmpty else {
            throw MLXRuntimeError.emptyPrompt
        }
        let input = try await container.prepare(input: UserInput(chat: messages))
        let parameters = GenerateParameters(
            maxTokens: 320,
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.05
        )
        return try await container.generate(input: input, parameters: parameters)
    }
}

enum MLXRuntimeError: LocalizedError {
    case incompletePack
    case emptyPrompt

    var errorDescription: String? {
        switch self {
        case .incompletePack:
            return "MLX folder is missing config.json, tokenizer.json, or the safetensors weights. Delete the pack and download it again."
        case .emptyPrompt:
            return "Nothing to send to the MLX model."
        }
    }
}
