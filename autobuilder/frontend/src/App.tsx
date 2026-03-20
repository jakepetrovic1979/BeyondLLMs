import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { OnboardingFlow } from './pages/Onboarding'
import { AppDashboard } from './pages/AppDashboard'
import { AppView } from './pages/AppView'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      cacheTime: 10 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

function AppIdWrapper({ component: Component }: { component: React.ComponentType<{ appId: string }> }) {
  const appId = window.location.pathname.split('/apps/')[1]?.split('/')[0]
  if (!appId) return <p>App not found</p>
  return <Component appId={appId} />
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
          <header style={{ padding: '16px 24px', borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', gap: 16 }}>
            <a href="/" style={{ fontSize: 20, fontWeight: 'bold', textDecoration: 'none', color: 'inherit' }}>AutoBuilder</a>
          </header>
          <main>
            <Routes>
              <Route path="/" element={<OnboardingFlow />} />
              <Route path="/apps/:appId" element={<AppIdWrapper component={AppDashboard} />} />
              <Route path="/apps/:appId/code" element={<AppIdWrapper component={AppView} />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
