import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { ImageUp, X } from 'lucide-react'

interface Props {
  onFileSelected: (file: File) => void
  previewUrl: string | null
  onClear: () => void
  disabled?: boolean
}

export function UploadZone({ onFileSelected, previewUrl, onClear, disabled }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (file && file.type.startsWith('image/')) {
        onFileSelected(file)
      }
    },
    [onFileSelected],
  )

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setIsDragging(false)
        if (!disabled) handleFiles(e.dataTransfer.files)
      }}
      onClick={() => !disabled && !previewUrl && inputRef.current?.click()}
      className="relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all duration-200 overflow-hidden"
      style={{
        borderColor: isDragging ? 'var(--brand)' : 'var(--border-strong)',
        background: isDragging
          ? 'color-mix(in oklab, var(--brand) 6%, var(--surface))'
          : 'var(--surface)',
        minHeight: previewUrl ? undefined : 280,
        cursor: disabled ? 'default' : previewUrl ? 'default' : 'pointer',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />

      {previewUrl ? (
        <div className="relative w-full">
          <img src={previewUrl} alt="Upload preview" className="w-full max-h-[420px] object-contain" />
          {!disabled && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onClear()
              }}
              aria-label="Remove image"
              className="cursor-pointer absolute top-3 right-3 flex items-center justify-center h-8 w-8 rounded-full shadow-lg backdrop-blur transition-transform hover:scale-105"
              style={{ background: 'rgba(11,11,11,0.65)' }}
            >
              <X size={16} color="#fff" />
            </button>
          )}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center gap-3 px-6 py-10 text-center"
        >
          <div
            className="flex items-center justify-center h-14 w-14 rounded-full"
            style={{ background: 'color-mix(in oklab, var(--brand) 12%, var(--surface))' }}
          >
            <ImageUp size={24} strokeWidth={1.5} style={{ color: 'var(--brand)' }} />
          </div>
          <div>
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
              Drop a chest X-ray or dermoscopy image
            </p>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              or click to browse &middot; PNG / JPEG &middot; modality is auto-detected
            </p>
          </div>
        </motion.div>
      )}
    </div>
  )
}
