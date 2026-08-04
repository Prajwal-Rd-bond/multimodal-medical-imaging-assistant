import { motion } from 'framer-motion'
import { FileText, Sparkles } from 'lucide-react'
import type { AnalyzeResponse } from '../lib/types'
import { ModalityBadge } from './ModalityBadge'
import { OodBanner } from './OodBanner'
import { ImageCompare } from './ImageCompare'
import { ConfidenceBar } from './ConfidenceBar'

interface Props {
  result: AnalyzeResponse
  originalUrl: string
}

export function ResultsPanel({ result, originalUrl }: Props) {
  const sorted = [...result.findings].sort((a, b) => b.calibrated_confidence - a.calibrated_confidence)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="grid gap-5 lg:grid-cols-2"
    >
      <div
        className="rounded-2xl border p-5"
        style={{ background: 'var(--surface)', borderColor: 'var(--border-strong)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <ModalityBadge modality={result.modality} />
        </div>
        <ImageCompare
          originalUrl={originalUrl}
          overlayUrl={result.overlay_image}
          overlayLabel={result.overlay_label}
          overlayConfident={result.overlay_confident}
        />
        <div className="mt-4">
          <OodBanner flagged={result.ood_flag} />
        </div>
      </div>

      <div className="flex flex-col gap-5">
        <div
          className="rounded-2xl border p-5"
          style={{ background: 'var(--surface)', borderColor: 'var(--border-strong)' }}
        >
          <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Calibrated confidence by finding
          </h3>
          <div className="flex flex-col gap-4">
            {sorted.map((f, i) => (
              <ConfidenceBar key={f.label} finding={f} delay={i * 0.05} />
            ))}
          </div>
        </div>

        <div
          className="rounded-2xl border p-5 flex-1"
          style={{ background: 'var(--surface)', borderColor: 'var(--border-strong)' }}
        >
          <h3 className="flex items-center gap-2 text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
            {result.used_llm ? (
              <Sparkles size={15} strokeWidth={2} style={{ color: 'var(--brand)' }} />
            ) : (
              <FileText size={15} strokeWidth={2} style={{ color: 'var(--text-muted)' }} />
            )}
            Structured report
            {!result.used_llm && (
              <span className="text-xs font-normal ml-auto" style={{ color: 'var(--text-muted)' }}>
                templated &mdash; no ANTHROPIC_API_KEY set
              </span>
            )}
          </h3>
          <p
            className="text-sm whitespace-pre-line leading-relaxed"
            style={{ color: 'var(--text-secondary)' }}
          >
            {result.report_text}
          </p>
        </div>
      </div>
    </motion.div>
  )
}
