import SwiftUI
import WebKit
import Photos

struct ContentView: View {
    @State private var serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? ""
    @State private var showSetup = false

    var body: some View {
        Group {
            if serverURL.isEmpty || showSetup {
                SetupView(serverURL: $serverURL, showSetup: $showSetup)
            } else {
                WebAppContainer(urlString: serverURL, showSetup: $showSetup)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            if serverURL.isEmpty { tryConnect() }
        }
    }

    func tryConnect() {
        let candidates = [
            "http://100.100.29.36:8080",
            "http://10.0.1.90:8080"
        ]
        for url in candidates {
            if let u = URL(string: url) {
                var req = URLRequest(url: u, timeoutInterval: 3)
                req.httpMethod = "HEAD"
                let sem = DispatchSemaphore(value: 0)
                var ok = false
                URLSession.shared.dataTask(with: req) { _, resp, _ in
                    if let http = resp as? HTTPURLResponse, http.statusCode < 500 { ok = true }
                    sem.signal()
                }.resume()
                sem.wait()
                if ok {
                    serverURL = url
                    UserDefaults.standard.set(url, forKey: "serverURL")
                    return
                }
            }
        }
    }
}

struct SetupView: View {
    @Binding var serverURL: String
    @Binding var showSetup: Bool
    @State private var input = ""

    var body: some View {
        ZStack {
            Color(red: 0.024, green: 0.039, blue: 0.071).ignoresSafeArea()
            VStack(spacing: 24) {
                Text("P")
                    .font(.system(size: 80, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(red: 0.494, green: 0.722, blue: 0.941))
                Text("PROMETHEON")
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .tracking(6)
                    .foregroundColor(Color(red: 0.494, green: 0.722, blue: 0.941).opacity(0.5))

                VStack(spacing: 12) {
                    TextField("Server URL", text: $input)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(12)
                        .background(Color.white.opacity(0.05))
                        .cornerRadius(8)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .onAppear { input = serverURL.isEmpty ? "http://100.100.29.36:8080" : serverURL }

                    Button(action: {
                        serverURL = input.trimmingCharacters(in: .whitespacesAndNewlines)
                        UserDefaults.standard.set(serverURL, forKey: "serverURL")
                        showSetup = false
                    }) {
                        Text("Connect")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .tracking(2)
                            .foregroundColor(.black)
                            .frame(maxWidth: .infinity)
                            .padding(12)
                            .background(Color(red: 0.494, green: 0.722, blue: 0.941))
                            .cornerRadius(8)
                    }
                }
                .padding(.horizontal, 40)
            }
        }
    }
}

// Loading bar that observes WKWebView.estimatedProgress
struct WebAppContainer: View {
    let urlString: String
    @Binding var showSetup: Bool
    @State private var progress: Double = 0
    @State private var isLoading = true

    var body: some View {
        ZStack(alignment: .top) {
            WebAppView(urlString: urlString, showSetup: $showSetup, progress: $progress, isLoading: $isLoading)
                .ignoresSafeArea()

            if isLoading {
                GeometryReader { geo in
                    Rectangle()
                        .fill(Color(red: 0.494, green: 0.722, blue: 0.941))
                        .frame(width: geo.size.width * progress, height: 3)
                        .shadow(color: Color(red: 0.494, green: 0.722, blue: 0.941).opacity(0.8), radius: 4)
                        .animation(.easeOut(duration: 0.2), value: progress)
                }
                .frame(height: 3)
            }
        }
    }
}

struct WebAppView: UIViewRepresentable {
    let urlString: String
    @Binding var showSetup: Bool
    @Binding var progress: Double
    @Binding var isLoading: Bool

    static let token = "d161410ad31e4029e250ddab3734c41949a439ec0967ddb6b66309047ca2692f"

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        config.processPool = WKProcessPool()

        // Auth patching — must run before page scripts
        let authJS = """
        window.__PROMETHEON_TOKEN = '\(WebAppView.token)';
        window.__IS_NATIVE_APP = true;

        const _origFetch = window.fetch;
        window.fetch = function(url, opts) {
            opts = opts || {};
            if (!opts.headers) opts.headers = {};
            if (typeof opts.headers === 'object' && !(opts.headers instanceof Headers)) {
                opts.headers['Authorization'] = 'Bearer ' + window.__PROMETHEON_TOKEN;
            }
            return _origFetch.call(this, url, opts);
        };

        const _origOpen = XMLHttpRequest.prototype.open;
        const _origSend = XMLHttpRequest.prototype.send;
        XMLHttpRequest.prototype.open = function() { this._url = arguments[1]; return _origOpen.apply(this, arguments); };
        XMLHttpRequest.prototype.send = function() {
            this.setRequestHeader('Authorization', 'Bearer ' + window.__PROMETHEON_TOKEN);
            return _origSend.apply(this, arguments);
        };
        """
        config.userContentController.addUserScript(
            WKUserScript(source: authJS, injectionTime: .atDocumentStart, forMainFrameOnly: false)
        )

        // Download override — must run AFTER page scripts define downloadSinglePhoto
        let downloadJS = """
        window.downloadSinglePhoto = async function(path) {
            // Show toast
            let toast = document.getElementById('_dl_toast');
            if (!toast) {
                toast = document.createElement('div');
                toast.id = '_dl_toast';
                toast.style.cssText = 'position:fixed;top:env(safe-area-inset-top,44px);left:50%;transform:translateX(-50%);z-index:99999;background:rgba(7,12,22,0.95);border:1px solid #0c1e30;color:#9ab5f5;font-family:Rajdhani,sans-serif;font-size:13px;font-weight:600;letter-spacing:0.1em;padding:10px 24px;border-radius:8px;backdrop-filter:blur(12px);transition:opacity 0.3s;pointer-events:none;text-transform:uppercase';
                document.body.appendChild(toast);
            }
            toast.textContent = 'Downloading...';
            toast.style.opacity = '1';

            const url = '/api/photos/download?p=' + encodeURIComponent(path);
            try {
                const r = await fetch(url);
                if (!r.ok) throw new Error(r.status);
                const blob = await r.blob();
                toast.textContent = 'Saving to Camera Roll...';
                const reader = new FileReader();
                reader.onloadend = function() {
                    window.webkit.messageHandlers.savePhoto.postMessage({
                        data: reader.result,
                        filename: path.split('/').pop()
                    });
                };
                reader.readAsDataURL(blob);
            } catch(e) {
                toast.textContent = 'Download failed';
                setTimeout(() => { toast.style.opacity = '0'; }, 2000);
            }
        };
        """
        config.userContentController.addUserScript(
            WKUserScript(source: downloadJS, injectionTime: .atDocumentEnd, forMainFrameOnly: false)
        )

        // Register message handler for saving photos
        config.userContentController.add(context.coordinator, name: "savePhoto")

        let wv = WKWebView(frame: .zero, configuration: config)
        wv.navigationDelegate = context.coordinator
        wv.uiDelegate = context.coordinator
        wv.scrollView.contentInsetAdjustmentBehavior = .automatic
        wv.isOpaque = false
        wv.backgroundColor = UIColor(red: 0.024, green: 0.039, blue: 0.071, alpha: 1)
        wv.scrollView.backgroundColor = wv.backgroundColor
        wv.allowsBackForwardNavigationGestures = true
        context.coordinator.webView = wv

        if let url = URL(string: urlString) {
            wv.load(URLRequest(url: url))
        }

        // Observe loading progress
        context.coordinator.progressObserver = wv.observe(\.estimatedProgress) { wv, _ in
            DispatchQueue.main.async {
                self.progress = wv.estimatedProgress
            }
        }
        context.coordinator.loadingObserver = wv.observe(\.isLoading) { wv, _ in
            DispatchQueue.main.async {
                self.isLoading = wv.isLoading
            }
        }

        return wv
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        var parent: WebAppView
        weak var webView: WKWebView?
        var progressObserver: NSKeyValueObservation?
        var loadingObserver: NSKeyValueObservation?

        init(_ parent: WebAppView) { self.parent = parent }

        // Handle native photo save from JS
        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == "savePhoto",
                  let body = message.body as? [String: String],
                  let dataURL = body["data"],
                  let filename = body["filename"] else { return }

            // Parse data URL: "data:image/jpeg;base64,..."
            guard let commaIdx = dataURL.firstIndex(of: ",") else { return }
            let base64 = String(dataURL[dataURL.index(after: commaIdx)...])
            guard let data = Data(base64Encoded: base64) else { return }

            let ext = (filename as NSString).pathExtension.lowercased()
            let videoExts = ["mp4", "mov", "m4v"]

            let showResult = { (ok: Bool) in
                DispatchQueue.main.async {
                    let js = ok
                        ? "document.getElementById('_dl_toast').textContent='Saved!';setTimeout(()=>{document.getElementById('_dl_toast').style.opacity='0'},1500)"
                        : "document.getElementById('_dl_toast').textContent='Save failed';setTimeout(()=>{document.getElementById('_dl_toast').style.opacity='0'},2000)"
                    self.webView?.evaluateJavaScript(js)
                }
            }

            if videoExts.contains(ext) {
                let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
                try? data.write(to: tmp)
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: tmp)
                }) { ok, _ in
                    try? FileManager.default.removeItem(at: tmp)
                    showResult(ok)
                }
            } else {
                guard let image = UIImage(data: data) else { showResult(false); return }
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                }) { ok, _ in
                    showResult(ok)
                }
            }
        }

        // Handle navigation
        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, preferences: WKWebpagePreferences, decisionHandler: @escaping (WKNavigationActionPolicy, WKWebpagePreferences) -> Void) {
            if let url = navigationAction.request.url,
               url.path.contains("/api/photos/download") {
                decisionHandler(.cancel, preferences)
                return
            }
            decisionHandler(.allow, preferences)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            let code = (error as NSError).code
            if code == NSURLErrorTimedOut || code == NSURLErrorCannotConnectToHost {
                parent.showSetup = true
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            let code = (error as NSError).code
            if code == NSURLErrorTimedOut || code == NSURLErrorCannotConnectToHost {
                parent.showSetup = true
            }
        }

        // JS alerts
        func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
            let vc = UIApplication.shared.connectedScenes
                .compactMap { ($0 as? UIWindowScene)?.keyWindow?.rootViewController }
                .first
            let ac = UIAlertController(title: nil, message: message, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler() })
            vc?.present(ac, animated: true) ?? completionHandler()
        }

        // JS confirm
        func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
            let vc = UIApplication.shared.connectedScenes
                .compactMap { ($0 as? UIWindowScene)?.keyWindow?.rootViewController }
                .first
            let ac = UIAlertController(title: nil, message: message, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(false) })
            ac.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler(true) })
            vc?.present(ac, animated: true) ?? completionHandler(false)
        }
    }
}
