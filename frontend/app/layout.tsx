import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Navbar } from "@/components/navbar";
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: "NeuroSeg AI — Brain Tumor Segmentation & Classification",
  description:
    "Upload a brain MRI to get U-Net tumor segmentation, ConvLSTM classification, confidence, Grad-CAM explainability and a downloadable report.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <Navbar />
          <main className="mx-auto min-h-[calc(100vh-4rem)] max-w-7xl px-4 py-8">
            {children}
          </main>
          <Toaster richColors position="top-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
