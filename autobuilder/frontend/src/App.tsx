import React from 'react'
import { QueryClient, QueryClientProvider } from 'react-query'
import { OnboardingFlow } from './pages/Onboarding'

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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
        <header style={{ padding: '16px 24px', borderBottom: '1px solid #e5e7eb' }}>
          <h1 style={{ fontSize: 20, fontWeight: 'bold' }}>AutoBuilder</h1>
        </header>
        <main>
          <OnboardingFlow />
        </main>
      </div>
    </QueryClientProvider>
  )
}

export default App
