import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Microscope, TriangleAlert } from 'lucide-react'
import { ThemeToggle } from './components/ThemeToggle'
import { DisclaimerBanner } from './components/DisclaimerBanner'
import { UploadZone } from './components/UploadZone'
import { ResultsPanel } from './components/ResultsPanel'
import { analyzeImage, checkHealth, ApiError } from './lib/api'
import type { AnalyzeResponse } from './lib/types'

type Status = 'idle' | 'analyzing' | 'done' | 'error'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const objectUrlRef = useRef<string | null>(null)

  useEffect(() => {
    checkHealth()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false))
  }, [])

  function handleFileSelected(f: File) {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(f)
    objectUrlRef.current = url
    setFile(f)
    setPreviewUrl(url)
    setResult(null)
    setError(null)
    setStatus('idle')
  }

  function handleClear() {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    objectUrlRef.current = null
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setStatus('idle')
  }

  async function handleAnalyze() {
    if (!file) return
    setStatus('analyzing')
    setError(null)
    try {
      const res = await analyzeImage(file)
      setResult(res)
      setStatus('done')
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong. Is the backend running?')
      setStatus('error')
    }
  }

  return (
    <div className="min-h-screen" style={{ background: 'var(--page)' }}>
      <header
        className="sticky top-0 z-10 backdrop-blur border-b"
        style={{ borderColor: 'var(--border)', background: 'color-mix(in oklab, var(--page) 80%, transparent)' }}
      >
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div
              className="flex items-center justify-center h-8 w-8 rounded-lg"
              style={{ background: 'var(--brand)' }}
            >
              <Microscope size={17} strokeWidth={2} color="var(--brand-ink)" />
            </div>
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              Multimodal Medical Imaging Assistant
            </span>
          </div>
          <div className="flex items-center gap-3">
            {backendUp === false && (
              <span
                className="hidden sm:inline-flex items-center gap-1.5 text-xs rounded-full px-2.5 py-1"
                style={{
                  color: 'var(--status-critical)',
                  background: 'color-mix(in oklab, var(--status-critical) 12%, transparent)',
                }}
              >
                <TriangleAlert size={12} /> backend unreachable
              </span>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 pb-24">
        <motion.section
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="pt-14 pb-8 text-center"
        >
          <h1
            className="text-4xl sm:text-5xl font-semibold tracking-tight text-balance"
            style={{ color: 'var(--text-primary)' }}
          >
            Chest X-ray &amp; dermoscopy analysis,
            <br />
            explained &amp; calibrated.
          </h1>
          <p className="mt-4 max-w-xl mx-auto text-base" style={{ color: 'var(--text-secondary)' }}>
            Upload an image. The pipeline auto-detects the modality, classifies findings,
            explains them with Grad-CAM, calibrates confidence, and flags anything
            out-of-distribution &mdash; before an LLM turns it into a structured report.
          </p>
        </motion.section>

        <div className="mb-6">
          <DisclaimerBanner />
        </div>

        <div className="flex flex-col gap-4">
          <UploadZone
            onFileSelected={handleFileSelected}
            previewUrl={previewUrl}
            onClear={handleClear}
            disabled={status === 'analyzing'}
          />

          {file && status !== 'done' && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              onClick={handleAnalyze}
              disabled={status === 'analyzing' || backendUp === false}
              className="cursor-pointer self-center flex items-center gap-2 rounded-full px-6 py-3 font-medium transition-transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100"
              style={{ background: 'var(--brand)', color: 'var(--brand-ink)' }}
            >
              {status === 'analyzing' && <Loader2 size={16} className="animate-spin" />}
              {status === 'analyzing' ? 'Analyzing…' : 'Analyze image'}
            </motion.button>
          )}

          {error && (
            <div
              className="flex items-center gap-2 rounded-lg px-4 py-3 text-sm self-center"
              style={{
                color: 'var(--status-critical)',
                background: 'color-mix(in oklab, var(--status-critical) 10%, transparent)',
              }}
            >
              <TriangleAlert size={16} />
              {error}
            </div>
          )}
        </div>

        {result && previewUrl && (
          <div className="mt-10">
            <ResultsPanel result={result} originalUrl={previewUrl} />
            <div className="mt-6">
              <DisclaimerBanner />
            </div>
          </div>
        )}
      </main>

      <footer className="max-w-5xl mx-auto px-6 pb-10 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
        Research demo &middot; NIH ChestX-ray14 &amp; ISIC training data &middot; not for clinical use
      </footer>
    </div>
  )
}

export default App
