/**
 * Edge middleware — gates the entire app behind the login cookie.
 *
 * Public routes (no auth needed):
 *   - /login
 *   - /api/auth/*   (login + logout endpoints)
 *   - Static assets are excluded via the `matcher` config below
 *
 * Anything else: if cookie missing or signature invalid/expired, redirect to
 * /login with the original path preserved as ?next=<path>.
 *
 * Set DTS_LOGIN_PASSWORD and DTS_AUTH_SECRET in Vercel project env vars.
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { COOKIE_NAME, readAuthEnv, verifyToken } from "@/lib/auth";

const PUBLIC_PATHS = ["/login", "/api/auth/login", "/api/auth/logout"];

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Public routes pass through
  if (PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"))) {
    return NextResponse.next();
  }

  const token = req.cookies.get(COOKIE_NAME)?.value;
  let secret: string;
  try {
    secret = readAuthEnv().secret;
  } catch (err) {
    // Misconfigured prod — fail closed
    return new NextResponse(
      `Authentication misconfigured: ${(err as Error).message}`,
      { status: 500 },
    );
  }

  const payload = await verifyToken(token, secret);
  if (payload) {
    return NextResponse.next();
  }

  // Not authenticated → redirect to /login (preserving destination)
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = pathname === "/" ? "" : `?next=${encodeURIComponent(pathname + req.nextUrl.search)}`;
  return NextResponse.redirect(url);
}

/**
 * Skip Next.js internals + static assets entirely. Anything matching this
 * pattern goes through `middleware()`, anything else is uninspected.
 */
export const config = {
  matcher: [
    // Match all request paths except those starting with:
    // - _next/static (static files)
    // - _next/image  (image optimization files)
    // - favicon.ico
    // - data/ (public data)
    // - fonts/ (public fonts)
    // - any file extension (e.g. .png, .jpg, .svg, .css, .js, .woff2)
    "/((?!_next/static|_next/image|favicon.ico|data/|fonts/|.*\\.(?:png|jpg|jpeg|gif|svg|webp|ico|css|js|woff2?|map)).*)",
  ],
};
