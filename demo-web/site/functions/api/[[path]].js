const TUNNEL = /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/;
const ALLOWED = /^(health|simplify|jobs\/[A-Za-z0-9_-]{1,32})$/;

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });

function same(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function onRequest({ request, env, params }) {
  const path = (params.path || []).join("/");
  if (path === "_register") {
    if (request.method !== "POST" || !same(request.headers.get("x-demo-key"), env.DEMO_SECRET)) {
      return json(403, { error: "forbidden" });
    }
    const { url } = await request.json().catch(() => ({}));
    if (!TUNNEL.test(url || "")) return json(400, { error: "bad url" });
    await env.DEMO_KV.put("tunnel", url);
    return json(200, { ok: true });
  }
  if (!ALLOWED.test(path)) return json(404, { error: "not found" });
  const method = path === "simplify" ? "POST" : "GET";
  if (request.method !== method) return json(405, { error: "method" });

  const tunnel = await env.DEMO_KV.get("tunnel", { cacheTtl: 30 });
  if (!tunnel) return json(503, { offline: true });
  let body;
  if (method === "POST") {
    body = await request.text();
    if (body.length > 16000) return json(413, { error: "too_long", max: 2000 });
  }
  try {
    const upstream = await fetch(`${tunnel}/api/${path}`, {
      method,
      body,
      headers: {
        "content-type": "application/json",
        "x-demo-key": env.DEMO_SECRET,
        "x-demo-client-ip": request.headers.get("cf-connecting-ip") || "",
      },
      signal: AbortSignal.timeout(20000),
    });
    const ct = upstream.headers.get("content-type") || "";
    if (!ct.includes("application/json") || upstream.status >= 500) return json(503, { offline: true });
    return json(upstream.status, await upstream.json());
  } catch {
    return json(503, { offline: true });
  }
}
