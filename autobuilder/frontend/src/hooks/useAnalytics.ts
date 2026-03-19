import { useCallback } from 'react'
import { appAPI } from '../services/api'

export function useAnalytics(appId: string) {
  const trackAction = useCallback(
    (actionType: string, actionData?: any) => {
      appAPI.recordAction(appId, {
        type: actionType,
        data: actionData,
      }).catch(console.error)
    },
    [appId]
  )

  return { trackAction }
}
