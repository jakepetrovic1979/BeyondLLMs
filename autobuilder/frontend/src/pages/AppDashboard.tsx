import React from 'react'
import { useQuery } from 'react-query'
import { appAPI } from '../services/api'

interface MetaStateProps {
  label: string
  value: number
  target: number
}

function MetricCard({ label, value, target }: MetaStateProps) {
  const percentage = (value / target) * 100
  const color = percentage > 80 ? '#22c55e' : percentage > 50 ? '#eab308' : '#ef4444'

  return (
    <div style={{ padding: 16, border: '1px solid #e5e7eb', borderRadius: 8, background: '#fafafa' }}>
      <p style={{ fontSize: 14, color: '#6b7280' }}>{label}</p>
      <p style={{ fontSize: 28, fontWeight: 'bold', color }}>{(value * 100).toFixed(1)}%</p>
      <p style={{ fontSize: 12, color: '#9ca3af' }}>Target: {(target * 100).toFixed(0)}%</p>
    </div>
  )
}

interface ImprovementData {
  id: string
  type: string
  description: string
  riskLevel: string
  status: string
}

function ImprovementCard({ improvement }: { improvement: ImprovementData }) {
  const riskColor =
    improvement.riskLevel === 'low' ? '#22c55e' :
    improvement.riskLevel === 'medium' ? '#eab308' : '#ef4444'

  const handleApprove = async () => {
    await appAPI.approveImprovement(improvement.id)
    window.location.reload()
  }

  const handleReject = async () => {
    await appAPI.rejectImprovement(improvement.id)
    window.location.reload()
  }

  return (
    <div style={{ padding: 16, border: '1px solid #e5e7eb', borderRadius: 8, marginTop: 16 }}>
      <h3 style={{ fontWeight: 600 }}>{improvement.description}</h3>
      <p style={{ fontSize: 14, color: '#6b7280', marginTop: 8 }}>Type: {improvement.type}</p>
      <p style={{ fontSize: 14, fontWeight: 600, color: riskColor, marginTop: 4 }}>
        Risk: {improvement.riskLevel.toUpperCase()}
      </p>
      <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
        <button
          onClick={handleApprove}
          style={{ padding: '8px 16px', background: '#22c55e', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
        >
          Approve
        </button>
        <button
          onClick={handleReject}
          style={{ padding: '8px 16px', background: '#9ca3af', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
        >
          Reject
        </button>
      </div>
    </div>
  )
}

export function AppDashboard({ appId }: { appId: string }) {
  const { data: appData, isLoading } = useQuery(
    ['app', appId],
    () => appAPI.getApp(appId).then((r) => r.data)
  )

  if (isLoading) return <p>Loading...</p>
  if (!appData) return <p>App not found</p>

  const metaState = appData.metaState
  const improvements = appData.improvements || []

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 'bold' }}>{appData.name}</h1>
          <p style={{ color: '#6b7280', marginTop: 4 }}>{appData.description}</p>
        </div>
        <a
          href={`/apps/${appId}/code`}
          style={{ padding: '8px 16px', background: '#2563eb', color: '#fff', borderRadius: 4, textDecoration: 'none', fontSize: 14 }}
        >
          View Code
        </a>
      </div>

      {/* Meta-States */}
      {metaState && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 24 }}>
          <MetricCard label="Prediction Error" value={1 - metaState.predictionError} target={0.8} />
          <MetricCard label="Integration (Phi)" value={metaState.phi} target={0.7} />
          <MetricCard label="Transparency" value={1 - metaState.opacity} target={0.8} />
        </div>
      )}

      {/* Suggested Improvements */}
      <div style={{ marginTop: 32 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600 }}>Suggested Improvements</h2>
        {improvements.length === 0 ? (
          <p style={{ color: '#9ca3af', marginTop: 8 }}>No pending improvements</p>
        ) : (
          improvements.map((imp: ImprovementData) => (
            <ImprovementCard key={imp.id} improvement={imp} />
          ))
        )}
      </div>

      {/* Deployment Status */}
      <div style={{ marginTop: 32 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600 }}>Deployment</h2>
        <p style={{ color: '#6b7280', marginTop: 8 }}>
          Status: <span style={{ fontWeight: 600 }}>{appData.deploymentStatus}</span>
        </p>
        {appData.deploymentUrl && (
          <a
            href={appData.deploymentUrl}
            target="_blank"
            rel="noreferrer"
            style={{ color: '#2563eb', marginTop: 4, display: 'inline-block' }}
          >
            {appData.deploymentUrl}
          </a>
        )}
      </div>
    </div>
  )
}
