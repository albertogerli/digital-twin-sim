/**
 * POST /api/auth/invite/create
 *
 * Body: { label?: string, expiresInDays?: number }   default 7
 * Returns: { ok: true, url, token, expiresAt, label, sub }
 *
 * Requires the caller to already be authenticated (valid session cookie).
 *
 * Edge runtime, Web Crypto only.
 */

import { NextResponse } from "next/server";
import {
  COOKIE_NAME,
  readAuthEnv,
  signToken,
  verifyToken,
} from "@/lib/auth";

export const runtime = "edge";

const DAY_SECONDS = 60 * 60 * 24;
const MAX_DAYS = 90;
const MIN_DAYS = 1;

function newId(): string {
  // 16 hex chars — 64 bits of entropy, plenty for a non-secret correlation id
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function POST(req: Request) {
  let auth: { password: string; secret: string };
  try {
    auth = readAuthEnv();
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: (err as Error).message },
      { status: 500 },
    );
  }

  // ── Caller must already be signed in ─────────────────────
  const cookieHeader = req.headers.get("cookie") || "";
  const cookieMatch = cookieHeader.match(new RegExp(`(?:^|; )${COOKIE_NAME}=([^;]+)`));
  const sessionToken = cookieMatch ? decodeURIComponent(cookieMatch[1]) : undefined;
  const session = await verifyToken(sessionToken, auth.secret, "session");
  if (!session) {
    return NextResponse.json(
      { ok: false, error: "Sign in to generate invites" },
      { status: 401 },
    );
  }

  // ── Body ─────────────────────────────────────────────────
  let body: { label?: string; expiresInDays?: number };
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const rawDays = Number(body?.expiresInDays ?? 7);
  const days = Math.max(MIN_DAYS, Math.min(MAX_DAYS, Math.floor(rawDays || 7)));
  const label = (body?.label ?? "").toString().trim().slice(0, 80);
  const sub = newId();
  const exp = Math.floor(Date.now() / 1000) + days * DAY_SECONDS;

  const token = await signToken(
    { kind: "invite", exp, sub, label },
    auth.secret,
  );

  // Build the absolute invite URL using the request's own origin so it works
  // identically on Vercel preview deploys, prod, and localhost.
  const proto = req.headers.get("x-forwarded-proto") ?? "https";
  const host = req.headers.get("x-forwarded-host") ?? req.headers.get("host") ?? "localhost:3000";
  const url = `${proto}://${host}/invite?t=${encodeURIComponent(token)}`;

  return NextResponse.json({
    ok: true,
    url,
    token,
    sub,
    label,
    expiresAt: new Date(exp * 1000).toISOString(),
  });
}
