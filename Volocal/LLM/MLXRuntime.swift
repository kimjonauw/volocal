import Foundation
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Loads an on-disk MLX pack and streams a chat reply.
/// The weights are a Hugging Face MLX folder, not a GGUF.
enum MLXRuntime {
    static func load(directory: URL) async throws -> ModelContainer {
        guard LLMModelSpec.mlxPackIsComplete(at: directory) else {
            throw MLXRuntimeError.incompletePack
        }
        return try await LLMModelFactory.shared.loadContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )
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
            return "MLX folder is missing config.json or the safetensors weights."
        case .emptyPrompt:
            return "Nothing to send to the MLX model."
        }
    }
}
