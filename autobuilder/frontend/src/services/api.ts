import axios from 'axios'

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
})

export const appAPI = {
  // App CRUD
  createApp: (data: {
    appName: string
    appDescription: string
    features?: string[]
    dataModel?: Record<string, string[]>
  }) => API.post('/generate', data),

  getApp: (appId: string) => API.get(`/apps/${appId}`),

  listApps: (userId?: string) =>
    API.get('/apps', { params: userId ? { userId } : {} }),

  // Analytics
  recordAction: (appId: string, action: { type: string; data?: any; error?: string }) =>
    API.post(`/apps/${appId}/analytics`, action),

  // Improvements
  getImprovements: (appId: string) =>
    API.get(`/apps/${appId}/improvements/suggested`),

  approveImprovement: (improvementId: string) =>
    API.post(`/improvements/${improvementId}/approve`),

  rejectImprovement: (improvementId: string) =>
    API.post(`/improvements/${improvementId}/reject`),

  // Health
  healthCheck: () => API.get('/health'),
}

export default API
