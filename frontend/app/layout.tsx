import type { Metadata } from "next";
import { Atkinson_Hyperlegible, Fraunces } from "next/font/google";
import { SiteHeader } from "@/components/site-header";
import { A11Y_BOOT_SCRIPT } from "@/lib/a11y";
import "./globals.css";

// Atkinson Hyperlegible was designed for readers with low vision.
const body = Atkinson_Hyperlegible({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const display = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Verity — Is this real?",
  description: "Protect your family from AI voice-cloning scams. Verify first, detect second.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: A11Y_BOOT_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <SiteHeader />
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 pt-8 pb-10 sm:px-6">{children}</main>
        <footer className="site-footer border-t border-border/70 px-4 pt-6 pb-28 text-center text-sm text-muted-foreground lg:pb-6">
          Verity gives a signal, not proof. When in doubt, hang up and call back on a number you trust.
        </footer>
      </body>
    </html>
  );
}
