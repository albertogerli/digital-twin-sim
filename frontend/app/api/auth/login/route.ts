/**
 * POST /api/auth/login
 *
 * Body: { password: string }
 * Returns: { ok: true } and sets HttpOnly auth cookie on success,
 *          { ok: false, error } on failure.
 *
 * Uses Edge runtime (Web Crypto only).
 */

import { NextResponse } from "next/server";
import {
  COOKIE_NAME,
  TOKEN_TTL_SECONDS,
  readAuthEnv,
  signToken,
} from "@/lib/auth";

export const runtime = "edge";

export async function POST(req: Request) {
  let body: { password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  const submitted = (body?.password ?? "").trim();
  if (!submitted) {
    return NextResponse.json({ ok: false, error: "Password required" }, { status: 400 });
  }

  let auth: { password: string; secret: string };
  try {
    auth = readAuthEnv();
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: (err as Error).message },
      { status: 500 },
    );
  }

  // Constant-time-ish password compare via Web Crypto subtle.timingSafeEqual
  // not available; use the same XOR trick from the verify path.
  const a = new TextEncoder().encode(submitted);
  const b = new TextEncoder().encode(auth.password);
  let ok = a.length === b.length;
  let diff = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    diff |= (a[i] ?? 0) ^ (b[i] ?? 0);
  }
  if (!ok || diff !== 0) {
    // Tiny artificial delay to slow brute-force attempts (real rate-limiting
    // belongs at the platform layer, e.g. Vercel WAF).
    await new Promise((r) => setTimeout(r, 250));
    return NextResponse.json(
      { ok: false, error: "Incorrect password" },
      { status: 401 },
    );
  }

  const exp = Math.floor(Date.now() / 1000) + TOKEN_TTL_SECONDS;
  const token = await signToken({ exp }, auth.secret);

  const res = NextResponse.json({ ok: true });
  res.cookies.set({
    name: COOKIE_NAME,
    value: token,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: TOKEN_TTL_SECONDS,
  });
  return res;
}
