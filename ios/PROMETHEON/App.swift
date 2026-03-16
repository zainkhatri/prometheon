import SwiftUI

@main
struct PROMETHEONApp: App {
    @State private var showLaunch = true

    var body: some Scene {
        WindowGroup {
            ZStack {
                ContentView()
                    .ignoresSafeArea()

                if showLaunch {
                    LaunchScreen()
                        .transition(.opacity)
                        .zIndex(1)
                }
            }
            .onAppear {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    withAnimation(.easeOut(duration: 0.5)) {
                        showLaunch = false
                    }
                }
            }
        }
    }
}

struct LaunchScreen: View {
    @State private var barWidth: CGFloat = 0
    @State private var glowOpacity: Double = 0.3

    var body: some View {
        ZStack {
            Color(red: 0.016, green: 0.027, blue: 0.051)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                Text("P")
                    .font(.system(size: 90, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(red: 0.494, green: 0.722, blue: 0.941))
                    .shadow(color: Color(red: 0.494, green: 0.722, blue: 0.941).opacity(glowOpacity), radius: 20)

                Text("PROMETHEON")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .tracking(8)
                    .foregroundColor(Color(red: 0.494, green: 0.722, blue: 0.941).opacity(0.4))

                // Loading bar
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white.opacity(0.05))
                        .frame(width: 200, height: 3)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color(red: 0.494, green: 0.722, blue: 0.941))
                        .frame(width: barWidth, height: 3)
                        .shadow(color: Color(red: 0.494, green: 0.722, blue: 0.941).opacity(0.6), radius: 6)
                }
                .padding(.top, 16)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.8)) {
                barWidth = 200
            }
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                glowOpacity = 0.8
            }
        }
    }
}
