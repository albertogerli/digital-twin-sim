"use client";

import ScrollReveal from "./ScrollReveal";

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  light?: boolean;
  id?: string;
}

export default function SectionHeader({ title, subtitle, id }: SectionHeaderProps) {
  return (
    <ScrollReveal className="mb-12">
      <div id={id} className="scroll-mt-20">
        <h2 className="font-display text-3xl md:text-5xl font-bold mb-4 text-gray-900">
          {title}
        </h2>
        {subtitle && (
          <p className="font-body text-lg md:text-xl max-w-3xl text-gray-500">
            {subtitle}
          </p>
        )}
        <div className="mt-4 w-20 h-1 rounded bg-blue-500" />
      </div>
    </ScrollReveal>
  );
}
