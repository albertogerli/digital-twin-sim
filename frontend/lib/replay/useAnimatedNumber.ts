import { useState, useEffect, useRef } from "react";

export function useAnimatedNumber(
  target: number,
  duration = 800,
): number {
  const [displayed, setDisplayed] = useState(target);
  const currentRef = useRef(target);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const start = currentRef.current;
    const delta = target - start;
    if (Math.abs(delta) < 0.5) {
      setDisplayed(target);
      currentRef.current = target;
      return;
    }
    const startTime = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value = Math.round(start + delta * eased);
      currentRef.current = value;
      setDisplayed(value);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [target, duration]);

  return displayed;
}
