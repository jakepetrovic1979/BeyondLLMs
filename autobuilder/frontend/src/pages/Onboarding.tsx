import React, { useState } from 'react'
import { appAPI } from '../services/api'

type Step = 'describe' | 'generating' | 'deployed'

interface GeneratedApp {
  id: string
  name: string
  deploymentUrl?: string
}

export function OnboardingFlow() {
  const [step, setStep] = useState<Step>('describe')
  const [appName, setAppName] = useState('')
  const [description, setDescription] = useState('')
  const [generatedApp, setGeneratedApp] = useState<GeneratedApp | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    setStep('generating')

    try {
      const result = await appAPI.createApp({
        userId: 'demo-user',
        appName,
        appDescription: description,
      })

      setGeneratedApp(result.data.app)
      setStep('deployed')
    } catch (err: any) {
      setError(err.response?.data?.error || err.message)
      setStep('describe')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 640, margin: '0 auto', padding: 32 }}>
      {step === 'describe' && (
        <>
          <h1 style={{ fontSize: 28, fontWeight: 'bold' }}>What app do you want to build?</h1>
          <p style={{ color: '#666', marginTop: 8 }}>Describe your app in plain English</p>

          {error && (
            <div style={{ background: '#fee', border: '1px solid #fcc', padding: 12, marginTop: 16, borderRadius: 4 }}>
              {error}
            </div>
          )}

          <input
            type="text"
            placeholder="e.g., Todo app, Habit tracker, Blog"
            value={appName}
            onChange={(e) => setAppName(e.target.value)}
            style={{ width: '100%', marginTop: 16, padding: 12, border: '1px solid #ccc', borderRadius: 4, fontSize: 16 }}
          />

          <textarea
            placeholder="Describe what your app does and what features it should have..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            style={{ width: '100%', marginTop: 16, padding: 12, border: '1px solid #ccc', borderRadius: 4, height: 128, fontSize: 16 }}
          />

          <button
            onClick={handleGenerate}
            disabled={!appName || !description || loading}
            style={{
              marginTop: 24,
              padding: '10px 24px',
              background: !appName || !description ? '#ccc' : '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              fontSize: 16,
              cursor: !appName || !description ? 'not-allowed' : 'pointer',
            }}
          >
            Generate App
          </button>
        </>
      )}

      {step === 'generating' && (
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: 24, fontWeight: 'bold' }}>Generating your app...</h2>
          <p style={{ color: '#666', marginTop: 8 }}>This usually takes 30-60 seconds</p>
          <ol style={{ textAlign: 'left', marginTop: 24, lineHeight: 2 }}>
            <li>Parsing your description</li>
            <li>Generating database schema</li>
            <li>Creating backend code</li>
            <li>Building frontend UI</li>
          </ol>
        </div>
      )}

      {step === 'deployed' && generatedApp && (
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: 24, fontWeight: 'bold' }}>Your app has been generated!</h2>
          <p style={{ color: '#666', marginTop: 8 }}>
            App ID: <code>{generatedApp.id}</code>
          </p>

          <div style={{ marginTop: 32, textAlign: 'left' }}>
            <p style={{ fontWeight: 600 }}>What happens next:</p>
            <ul style={{ lineHeight: 2, marginTop: 8 }}>
              <li>Your app starts collecting usage data</li>
              <li>System M analyzes user behavior</li>
              <li>Improvements are suggested automatically</li>
              <li>You approve/reject changes</li>
              <li>App improves over time</li>
            </ul>
          </div>

          <button
            onClick={() => window.location.href = `/apps/${generatedApp.id}`}
            style={{
              marginTop: 32,
              padding: '10px 24px',
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              fontSize: 16,
              cursor: 'pointer',
            }}
          >
            Go to Dashboard
          </button>
        </div>
      )}
    </div>
  )
}
