import type { AnalyzeResponse, HealthResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export class ApiError extends Error {}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new ApiError('Backend is not reachable')
  return res.json()
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new ApiError(detail?.detail ?? `Analysis failed (${res.status})`)
  }

  return res.json()
}
