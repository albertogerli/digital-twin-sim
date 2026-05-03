/**
 * Edge-compatible HMAC-SHA256 token signing & verification.
 *
 * Token format: `<base64url(payload_json)>.<base64url(hmac_sig)>`
 * Payload shape: { exp: number }   (Unix seconds)
 *
 * Uses Web Crypto only (no Node imports), so this module runs unchanged in
 * Next.js middleware (Edge runtime) AND in API route handlers (Node runtime).
 */

export const COOKIE_NAME = "dts_auth";
export const TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

/**
 * `kind` discriminates session cookies from invite-link tokens so the two
 * verifiers can refuse to accept the wrong token type. Existing session
 * cookies signed before this field was added are treated as `"session"`.
 */
type TokenKind = "session" | "invite";

interface Payload {
  exp: number;
  kind?: TokenKind;
  sub?: string;       // session: tenant identifier · invite: invite id (uuid)
  label?: string;     // human label (e.g. "Marco Rossi") — shown in welcome
  isAdmin?: boolean;  // true for the password-login session, false for invite redemptions
}

/** Public type re-exported for the API routes that read identity from cookies. */
export type SessionPayload = Payload;

function b64urlEncode(bytes: ArrayBuffer | Uint8Array): string {
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let bin = "";
  for (let i = 0; i < arr.length; i++) bin += String.fromCharCode(arr[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function b64urlDecode(s: string): Uint8Array {
  const pad = (4 - (s.length % 4)) % 4;
  const b64 = (s + "=".repeat(pad)).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function getKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

/** Compare two strings in constant time to avoid timing attacks. */
function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

/** Sign a payload and return the encoded token string. */
export async function signToken(payload: Payload, secret: string): Promise<string> {
  const body = b64urlEncode(new TextEncoder().encode(JSON.stringify(payload)));
  const key = await getKey(secret);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body));
  return `${body}.${b64urlEncode(sig)}`;
}

/**
 * Verify token string. Returns the payload if signature is valid AND not expired
 * AND (when expectedKind is provided) the payload's kind matches.
 *
 * `expectedKind` defaults to `"session"` for backward compatibility — old
 * cookies without a kind field are treated as session cookies.
 */
export async function verifyToken(
  token: string | undefined,
  secret: string,
  expectedKind: TokenKind = "session",
): Promise<Payload | null> {
  if (!token) return null;
  const dot = token.indexOf(".");
  if (dot < 1) return null;
  const body = token.slice(0, dot);
  const sigGiven = token.slice(dot + 1);
  let sigGivenBytes: Uint8Array;
  try {
    sigGivenBytes = b64urlDecode(sigGiven);
  } catch {
    return null;
  }
  const key = await getKey(secret);
  const sigExpected = new Uint8Array(
    await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body)),
  );
  if (!timingSafeEqual(sigGivenBytes, sigExpected)) return null;
  let payload: Payload;
  try {
    const json = new TextDecoder().decode(b64urlDecode(body));
    payload = JSON.parse(json) as Payload;
  } catch {
    return null;
  }
  if (typeof payload.exp !== "number") return null;
  if (payload.exp < Math.floor(Date.now() / 1000)) return null;
  // kind defaults to "session" when missing, so legacy cookies still work
  const kind: TokenKind = payload.kind ?? "session";
  if (kind !== expectedKind) return null;
  return payload;
}

/**
 * Read auth secrets from env. In dev, sane fallbacks let `npm run dev` work
 * without setup; in prod (NODE_ENV=production), missing values are fatal.
 */
export function readAuthEnv(): { password: string; secret: string } {
  const password = process.env.DTS_LOGIN_PASSWORD || "";
  const secret = process.env.DTS_AUTH_SECRET || "";
  if (process.env.NODE_ENV === "production") {
    if (!password || !secret) {
      throw new Error(
        "DTS_LOGIN_PASSWORD and DTS_AUTH_SECRET must be set in production. " +
          "See README for setup.",
      );
    }
  } else {
    // Dev fallback — never use these in prod
    return {
      password: password || "dts-dev-pw",
      secret: secret || "dts-dev-secret-do-not-use-in-production",
    };
  }
  return { password, secret };
}
