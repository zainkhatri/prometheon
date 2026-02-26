// PROMETHEON Service Worker — single-bundle cache strategy
// Downloads ALL thumbnails in one streaming request, stores in Cache API.
// Every subsequent visit: cache-first, zero network for images.

const CACHE = 'prometheon-thumbs-v3';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

// Cache-first for thumbnails — instant if cached, fall back to network
self.addEventListener('fetch', event => {
    const { pathname } = new URL(event.request.url);
    if (!pathname.startsWith('/static/thumbs')) return;
    event.respondWith(
        caches.open(CACHE).then(cache =>
            cache.match(event.request).then(hit => {
                if (hit) return hit;
                return fetch(event.request, { credentials: 'same-origin' }).then(res => {
                    if (res.ok) cache.put(event.request, res.clone());
                    return res;
                });
            })
        )
    );
});

self.addEventListener('message', event => {
    if (event.data?.type === 'PRECACHE_BUNDLE') {
        precacheBundle(event.data.expectedCount || 0);
    }
});

async function precacheBundle(expectedCount) {
    const cache = await caches.open(CACHE);

    // Skip if already (mostly) cached
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

    const concat = (a, b) => {
        const c = new Uint8Array(a.length + b.length);
        c.set(a); c.set(b, a.length); return c;
    };
    const u16 = (b, i) => (b[i] << 8) | b[i + 1];
    const u32 = (b, i) => ((b[i] << 24) | (b[i+1] << 16) | (b[i+2] << 8) | b[i+3]) >>> 0;

    while (true) {
        let done, value;
        try { ({ done, value } = await reader.read()); } catch (e) { break; }
        if (done) break;

        // Only concat the small residual (< 1 thumbnail) with the new chunk
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

            puts.push(
                cache.put(
                    new URL(url, self.location.origin).href,
                    new Response(imgData, {
                        status: 200,
                        headers: {
                            'Content-Type': 'image/jpeg',
                            'Cache-Control': 'public, max-age=604800, immutable',
                        },
                    })
                )
            );

            // Flush every 100 puts to keep memory bounded
            if (puts.length >= 100) { await Promise.all(puts); puts = []; }
        }

        residual = chunk.slice(pos); // at most 1 thumbnail's worth of bytes
    }

    if (puts.length) await Promise.all(puts);
}
