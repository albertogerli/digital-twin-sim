import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import ErrorBoundary from "@/components/ui/ErrorBoundary";

// Suppress console.error noise from React's error boundary logging
const originalConsoleError = console.error;
beforeEach(() => {
  console.error = vi.fn();
});

afterEach(() => {
  console.error = originalConsoleError;
});

function ThrowingChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error("Test explosion");
  }
  return <div>Child content</div>;
}

describe("ErrorBoundary", () => {
  it("renders children when there is no error", () => {
    render(
      <ErrorBoundary>
        <div>Hello world</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("shows default fallback UI when a child throws", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Test explosion")).toBeInTheDocument();
  });

  it("shows custom fallback when provided", () => {
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Custom fallback")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  it("renders a Retry button in default fallback", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("Retry button resets error state and re-renders children", () => {
    // We need a component that can switch between throwing and not
    let shouldThrow = true;
    function ConditionalThrower() {
      if (shouldThrow) throw new Error("Boom");
      return <div>Recovered</div>;
    }

    const { rerender } = render(
      <ErrorBoundary>
        <ConditionalThrower />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();

    // Stop throwing before clicking retry
    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(screen.getByText("Recovered")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  it("calls onError callback when an error is caught", () => {
    const onError = vi.fn();
    render(
      <ErrorBoundary onError={onError}>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Test explosion" }),
      expect.objectContaining({ componentStack: expect.any(String) }),
    );
  });

  it("logs error to console.error", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(console.error).toHaveBeenCalled();
  });

  it("shows generic message when error has no message", () => {
    function NoMessageThrower(): JSX.Element {
      throw new Error();
    }

    render(
      <ErrorBoundary>
        <NoMessageThrower />
      </ErrorBoundary>,
    );
    expect(screen.getByText("An unexpected error occurred")).toBeInTheDocument();
  });
});
