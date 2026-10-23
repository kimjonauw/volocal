import SwiftUI

/// Always-visible compact debug overlay showing real-time system metrics.
/// Designed as a slim top pill that avoids overlapping navigation bar buttons.
struct MetricsOverlay: View {
    @EnvironmentObject var metrics: SystemMetrics
    @State private var isExpanded = false

    var body: some View {
        Button {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 5) {
                if isExpanded {
                    Label("\(metrics.memoryMB, specifier: "%.0f")/\(metrics.memoryLimitMB, specifier: "%.0f") MB", systemImage: "memorychip")
                        .foregroundStyle(memoryColor)
                    Text("·")
                        .foregroundStyle(.secondary)
                    Label("\(metrics.cpuPercent, specifier: "%.0f")%", systemImage: "cpu")
                        .foregroundStyle(.secondary)
                    Text("·")
                        .foregroundStyle(.secondary)
                    Text(metrics.thermalStateLabel)
                        .foregroundStyle(thermalColor)
                } else {
                    Image(systemName: "memorychip")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(memoryColor)
                    Text("\(Int(metrics.memoryMB))/\(Int(metrics.memoryLimitMB))M")
                        .foregroundStyle(memoryColor)
                    Text("·")
                        .foregroundStyle(.secondary)
                    Text("\(Int(metrics.cpuPercent))%")
                        .foregroundStyle(.secondary)
                    Circle()
                        .fill(thermalColor)
                        .frame(width: 5, height: 5)
                }
            }
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(Color.primary.opacity(0.12), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.08), radius: 3, y: 1)
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 4)
    }

    private var memoryColor: Color {
        let free = metrics.memoryLimitMB - metrics.memoryMB
        if free < 80 { return .red }
        if free < 180 { return .orange }
        return .primary
    }

    private var thermalColor: Color {
        switch metrics.thermalState {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }
}
