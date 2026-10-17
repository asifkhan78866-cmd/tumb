"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, X } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function UploadDropzone({
  onFile,
  disabled,
  label,
  hint,
  compact,
}: {
  onFile: (file: File | null) => void;
  disabled?: boolean;
  /** Headline inside the empty dropzone; defaults to the MRI wording. */
  label?: string;
  hint?: string;
  /** Shorter box, for a secondary image beside the main one. */
  compact?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setFileName(file.name);
      setPreview(URL.createObjectURL(file));
      onFile(file);
    },
    [onFile]
  );

  const clear = () => {
    setPreview(null);
    setFileName("");
    if (inputRef.current) inputRef.current.value = "";
    onFile(null);  // the parent must forget the file too, not just the preview
  };

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed text-center transition-colors",
          compact ? "min-h-[150px] p-4" : "min-h-[280px] p-8",
          dragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/40",
          disabled && "pointer-events-none opacity-60"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        {preview ? (
          <div className="relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt={label ? `${label} preview` : "MRI preview"}
              className={cn("mx-auto rounded-lg object-contain shadow-md",
                compact ? "max-h-28" : "max-h-48")}
            />
            <p className="mt-3 text-sm text-muted-foreground">{fileName}</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={(e) => {
                e.stopPropagation();
                clear();
              }}
            >
              <X className="h-4 w-4" /> Remove
            </Button>
          </div>
        ) : (
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex flex-col items-center gap-3"
          >
            <div
              className={cn(
                "flex items-center justify-center rounded-full bg-primary/10 text-primary",
                compact ? "h-10 w-10" : "h-16 w-16"
              )}
            >
              <UploadCloud className={compact ? "h-5 w-5" : "h-8 w-8"} />
            </div>
            <div>
              <p className={cn("font-medium", compact && "text-sm")}>
                {label ?? "Drag & drop an MRI image, or click to browse"}
              </p>
              <p className={cn("mt-1 text-muted-foreground", compact ? "text-xs" : "text-sm")}>
                {hint ?? "PNG, JPG or TIFF — a single brain MRI slice"}
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
