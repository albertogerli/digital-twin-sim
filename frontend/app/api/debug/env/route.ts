/**
 * GET /api/debug/env  (PUBLIC — safe: returns names only, never values)
 *
 * Diagnostic for Vercel env-var misconfiguration. Lists only the NAMES
 * of every env var visible to the Edge runtime that starts with `DTS_`,
 * plus a few generic Vercel/Node ones. Never echoes values.
 *
 * Use this when you see "Missing env var: DTS_LOGIN_PASSWORD" and want
 * to verify what's actually visible at runtime — typos, wrong scope,
 * missing redeploy all become obvious.
 *
 * Hit it directly:
 *   https://<your-deploy>.vercel.app/api/debug/env
 *
 * Once everything works, this route stays harmless (it leaks no
 * secrets), but feel free to delete it.
 */

import { NextResponse } from "next/server";

export const runtime = "edge";

export async function GET() {
  // Edge runtime exposes process.env as a normal object even though it's
  // not the Node Process. Iterate over it to get the visible keys.
  const allKeys = Object.keys(process.env || {});
  const dtsKeys = allKeys.filter((k) => k.startsWith("DTS_")).sort();

  // Also report whether each REQUIRED var is present (boolean only)
  const required = {
    DTS_LOGIN_PASSWORD: !!process.env.DTS_LOGIN_PASSWORD,
    DTS_AUTH_SECRET:    !!process.env.DTS_AUTH_SECRET,
  };

  // Length-only for the present ones — helps spot accidental empty/whitespace values
  const lengths: Record<string, number> = {};
  for (const k of Object.keys(required)) {
    const v = process.env[k];
    if (typeof v === "string") lengths[k] = v.length;
  }

  return NextResponse.json({
    runtime: "edge",
    nodeEnv: process.env.NODE_ENV ?? null,
    vercelEnv: process.env.VERCEL_ENV ?? null,        // "production" | "preview" | "development"
    deploymentUrl: process.env.VERCEL_URL ?? null,    // hostname of this deploy
    dtsKeysVisible: dtsKeys,                          // names only, never values
    requiredPresent: required,
    requiredValueLengths: lengths,                    // 0 = empty/whitespace-only
    totalEnvKeys: allKeys.length,
  });
}
