// PROMETHEON Service Worker — offline-capable with layered caching
const THUMB_CACHE = 'prometheon-thumbs-v8';
const PAGE_CACHE = 'prometheon-pages-v1';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(
    caches.keys().then(names =>
        Promise.all(names.filter(n => n !== THUMB_CACHE && n !== PAGE_CACHE).map(n => caches.delete(n)))
    ).then(() => self.clients.claim())
));

self.addEventListener('fetch', event => {
    const { pathname } = new URL(event.request.url);

    // ── Thumbnails & previews: cache-first (immutable images) ──
    if (pathname.startsWith('/static/thumbs')) {
        event.respondWith(
            caches.open(THUMB_CACHE).then(cache =>
                cache.match(event.request).then(hit => {
                    if (hit) return hit;
                    return fetch(event.request, { credentials: 'same-origin' }).then(res => {
                        if (res.ok) cache.put(event.request, res.clone());
                        return res;
                    }).catch(() => {
                        // Offline fallback: if requesting preview, try HQ thumb
                        if (pathname.includes('thumbs_preview')) {
                            const hqPath = pathname.replace('thumbs_preview', 'thumbs_hq');
                            return cache.match(new URL(hqPath, event.request.url).href)
                                .then(hq => hq || new Response('', { status: 503 }));
                        }
                        return new Response('', { status: 503 });
                    });
                })
            )
        );
        return;
    }

    // ── Journal page renders: cache-first ──
    if (pathname.includes('/api/journals/') && pathname.includes('/page/')) {
        event.respondWith(
            caches.open(THUMB_CACHE).then(cache =>
                cache.match(event.request).then(hit => {
                    if (hit) return hit;
                    return fetch(event.request, { credentials: 'same-origin' }).then(res => {
                        if (res.ok) cache.put(event.request, res.clone());
                        return res;
                    }).catch(() => new Response('', { status: 503 }));
                })
            )
        );
        return;
    }

    // ── HTML pages: network-first, cache fallback (offline browsing) ──
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request).then(res => {
                if (res.ok) {
                    const clone = res.clone();
                    caches.open(PAGE_CACHE).then(c => c.put(event.request, clone));
                }
                return res;
            }).catch(() => caches.open(PAGE_CACHE).then(c => c.match(event.request)))
        );
        return;
    }

    // ── API JSON: stale-while-revalidate ──
    if (pathname.startsWith('/api/')) {
        event.respondWith(
            caches.open(PAGE_CACHE).then(cache =>
                cache.match(event.request).then(cached => {
                    const networkFetch = fetch(event.request, { credentials: 'same-origin' }).then(res => {
                        if (res.ok) cache.put(event.request, res.clone());
                        return res;
                    }).catch(() => null);
                    // Return cached immediately if available, update in background
                    return cached || networkFetch || new Response('{"error":"offline"}', {
                        status: 503, headers: { 'Content-Type': 'application/json' }
                    });
                })
            )
        );
        return;
    }

    // ── Static assets (CSS, JS, fonts): cache-first ──
    if (pathname.startsWith('/static/') || pathname.includes('fonts.googleapis') || pathname.includes('cdn.tailwindcss')) {
        event.respondWith(
            caches.open(PAGE_CACHE).then(cache =>
                cache.match(event.request).then(hit => {
                    if (hit) return hit;
                    return fetch(event.request).then(res => {
                        if (res.ok) cache.put(event.request, res.clone());
                        return res;
                    }).catch(() => new Response('', { status: 503 }));
                })
            )
        );
        return;
    }
});

// ── Thumb bundle pre-cache (bulk download on first visit) ──
self.addEventListener('message', event => {
    if (event.data?.type === 'PRECACHE_BUNDLE') {
        precacheBundle(event.data.expectedCount || 0);
    }
});

async function precacheBundle(expectedCount) {
    const cache = await caches.open(THUMB_CACHE);
    const existing = await cache.keys();
    if (expectedCount > 0 && existing.length >= Math.floor(expectedCount * 0.95)) return;

    let resp;
    try {
        resp = await fetch('/api/photos/thumb-bundle', { credentials: 'same-origin' });
        if (!resp.ok || !resp.body) return;
    } catch (e) { return; }

    const reader = resp.body.getReader();
    let residual = new Uint8Array(0);
    let puts = [];

    const concat = (a, b) => { const c = new Uint8Array(a.length + b.length); c.set(a); c.set(b, a.length); return c; };
    const u16 = (b, i) => (b[i] << 8) | b[i + 1];
    const u32 = (b, i) => ((b[i] << 24) | (b[i+1] << 16) | (b[i+2] << 8) | b[i+3]) >>> 0;

    while (true) {
        let done, value;
        try { ({ done, value } = await reader.read()); } catch (e) { break; }
        if (done) break;
        const chunk = residual.length ? concat(residual, value) : value;
        let pos = 0;
        while (pos + 6 <= chunk.length) {
            const urlLen = u16(chunk, pos);
            const dataOffset = pos + 2 + urlLen;
            if (dataOffset + 4 > chunk.length) break;
            const dataLen = u32(chunk, dataOffset);
            const entryEnd = dataOffset + 4 + dataLen;
            if (entryEnd > chunk.length) break;
            const url = new TextDecoder().decode(chunk.slice(pos + 2, pos + 2 + urlLen));
            const imgData = chunk.slice(dataOffset + 4, entryEnd);
            pos = entryEnd;
            puts.push(cache.put(
                new URL(url, self.location.origin).href,
                new Response(imgData, { status: 200, headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'public, max-age=604800, immutable' } })
            ));
            if (puts.length >= 100) { await Promise.all(puts); puts = []; }
        }
        residual = chunk.slice(pos);
    }
    if (puts.length) await Promise.all(puts);
}
