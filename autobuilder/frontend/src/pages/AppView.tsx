import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { appAPI } from '../services/api'

type Tab = 'frontend' | 'backend' | 'database'

function CodeBlock({ code }: { code: string }) {
  return (
    <pre style={{
      background: '#1e1e1e',
      color: '#d4d4d4',
      padding: 24,
      borderRadius: 8,
      overflow: 'auto',
      fontSize: 14,
      lineHeight: 1.6,
      maxHeight: '70vh',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
    }}>
      {code}
    </pre>
  )
}

export function AppView({ appId }: { appId: string }) {
  const [tab, setTab] = useState<Tab>('frontend')

  const { data: app, isLoading } = useQuery(
    ['app', appId],
    () => appAPI.getApp(appId).then((r) => r.data)
  )

  if (isLoading) return <p style={{ padding: 24 }}>Loading...</p>
  if (!app) return <p style={{ padding: 24 }}>App not found</p>

  const tabs: { key: Tab; label: string }[] = [
    { key: 'frontend', label: 'Frontend' },
    { key: 'backend', label: 'Backend' },
    { key: 'database', label: 'Database Schema' },
  ]

  const codeMap: Record<Tab, string> = {
    frontend: app.frontendCode || 'No frontend code generated',
    backend: app.backendCode || 'No backend code generated',
    database: app.databaseSchema || 'No database schema generated',
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 'bold' }}>{app.name}</h1>
          <p style={{ color: '#6b7280', marginTop: 4 }}>{app.description}</p>
        </div>
        <a
          href={`/apps/${appId}`}
          style={{ padding: '8px 16px', background: '#2563eb', color: '#fff', borderRadius: 4, textDecoration: 'none', fontSize: 14 }}
        >
          Dashboard
        </a>
      </div>

      <div style={{ display: 'flex', gap: 0, marginTop: 24, borderBottom: '2px solid #e5e7eb' }}>
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              padding: '10px 20px',
              border: 'none',
              background: 'none',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: tab === t.key ? 600 : 400,
              color: tab === t.key ? '#2563eb' : '#6b7280',
              borderBottom: tab === t.key ? '2px solid #2563eb' : '2px solid transparent',
              marginBottom: -2,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <CodeBlock code={codeMap[tab]} />
      </div>
    </div>
  )
}
