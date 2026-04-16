"use client";

import { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
}

interface State {
  hasError: boolean;
}

export default class ChartErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2 text-center">
          <p className="text-xs text-ki-on-surface-muted">
            {this.props.fallbackLabel || "Chart non disponibile"}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
