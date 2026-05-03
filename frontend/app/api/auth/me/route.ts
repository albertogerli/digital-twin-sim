/**
 * GET /api/auth/me
 *
 * Returns the authenticated user's identity from the session cookie.
 *
 * Response:
 *   200 { ok: true, sub, label, isAdmin, expiresAt }
 *   401 { ok: false }   (no cookie / invalid / expired)
 *
 * Edge runtime, Web Crypto only.
 */

import { NextResponse } from "next/server";
import { COOKIE_NAME, readAuthEnv, verifyToken } from "@/lib/auth";

export const runtime = "edge";

export async function GET(req: Request) {
  let auth: { password: string; secret: string };
  try {
    auth = readAuthEnv();
  } catch (err) {
    return NextResponse.json({ ok: false, error: (err as Error).message }, { status: 500 });
  }

  const cookieHeader = req.headers.get("cookie") || "";
  const m = cookieHeader.match(new RegExp(`(?:^|; )${COOKIE_NAME}=([^;]+)`));
  const token = m ? decodeURIComponent(m[1]) : undefined;
  const session = await verifyToken(token, auth.secret, "session");

  if (!session) {
    return NextResponse.json({ ok: false }, { status: 401 });
  }

  return NextResponse.json({
    ok: true,
    sub:     session.sub ?? "anonymous",
    label:   session.label ?? "",
    isAdmin: !!session.isAdmin,
    expiresAt: new Date(session.exp * 1000).toISOString(),
  });
}
