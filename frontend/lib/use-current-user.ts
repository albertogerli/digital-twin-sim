"use client";

import { useEffect, useState } from "react";

/**
 * Identity returned by /api/auth/me. Cached process-wide so multiple
 * components can call useCurrentUser() without re-fetching.
 */
export interface CurrentUser {
  sub: string;
  label: string;
  isAdmin: boolean;
  expiresAt: string;
}

let _cache: CurrentUser | null | undefined; // undefined = not fetched yet
let _inflight: Promise<CurrentUser | null> | null = null;

async function fetchMe(): Promise<CurrentUser | null> {
  if (_cache !== undefined) return _cache;
  if (_inflight) return _inflight;
  _inflight = (async () => {
    try {
      const res = await fetch("/api/auth/me", { credentials: "same-origin" });
      if (!res.ok) {
        _cache = null;
        return null;
      }
      const data = await res.json();
      _cache = {
        sub: data.sub,
        label: data.label,
        isAdmin: !!data.isAdmin,
        expiresAt: data.expiresAt,
      };
      return _cache;
    } catch {
      _cache = null;
      return null;
    } finally {
      _inflight = null;
    }
  })();
  return _inflight;
}

/** Subscribe to the current user. Returns `undefined` while loading,
 *  `null` if unauthenticated, or the full user object once available. */
export function useCurrentUser(): CurrentUser | null | undefined {
  const [user, setUser] = useState<CurrentUser | null | undefined>(_cache);
  useEffect(() => {
    if (_cache !== undefined) {
      setUser(_cache);
      return;
    }
    let cancelled = false;
    fetchMe().then((u) => {
      if (!cancelled) setUser(u);
    });
    return () => { cancelled = true; };
  }, []);
  return user;
}

/** Reset the cache (call after logout so the next page load re-fetches). */
export function clearCurrentUserCache() {
  _cache = undefined;
  _inflight = null;
}
