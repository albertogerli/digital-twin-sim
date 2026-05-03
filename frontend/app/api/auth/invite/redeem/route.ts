/**
 * POST /api/auth/invite/redeem  (public)
 *
 * Body: { token: string }
 * Returns: { ok: true, label } and sets the standard session cookie.
 *
 * The invite token's signature + expiry is verified; on success we mint a
 * fresh session token (kind="session") with a 7-day TTL so the recipient
 * gets the same access surface as a password login.
 *
 * Edge runtime, Web Crypto only.
 */

import { NextResponse } from "next/server";
import {
  COOKIE_NAME,
  TOKEN_TTL_SECONDS,
  readAuthEnv,
  signToken,
  verifyToken,
} from "@/lib/auth";

export const runtime = "edge";

export async function POST(req: Request) {
  let body: { token?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  const token = (body?.token ?? "").trim();
  if (!token) {
    return NextResponse.json({ ok: false, error: "Token required" }, { status: 400 });
  }

  let auth: { password: string; secret: string };
  try {
    auth = readAuthEnv();
  } catch (err) {
    return NextResponse.json({ ok: false, error: (err as Error).message }, { status: 500 });
  }

  const invite = await verifyToken(token, auth.secret, "invite");
  if (!invite) {
    return NextResponse.json(
      { ok: false, error: "Invite link is invalid or has expired" },
      { status: 401 },
    );
  }

  const exp = Math.floor(Date.now() / 1000) + TOKEN_TTL_SECONDS;
  const session = await signToken(
    {
      kind: "session",
      exp,
      // Carry the invite's identity into the new session so per-user filtering
      // (X-Tenant-Id header → backend tenant_id) works transparently.
      sub: invite.sub ?? "guest",
      label: invite.label ?? "",
      isAdmin: false,
    },
    auth.secret,
  );

  const res = NextResponse.json({ ok: true, label: invite.label ?? "" });
  res.cookies.set({
    name: COOKIE_NAME,
    value: session,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: TOKEN_TTL_SECONDS,
  });
  return res;
}
