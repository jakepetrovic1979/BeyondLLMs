import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import { prisma } from './db'
import appsRouter from './api/apps'
import generationRouter from './api/generation'
import improvementsRouter from './api/improvements'
import healthRouter from './api/health'

dotenv.config()

const app = express()
app.use(cors())
app.use(express.json({ limit: '10mb' }))

// Routes
app.use('/api', healthRouter)
app.use('/api', appsRouter)
app.use('/api', generationRouter)
app.use('/api', improvementsRouter)

const PORT = process.env.PORT || 3001

async function main() {
  // Start listening first so Railway healthcheck can reach us
  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`AutoBuilder API running on port ${PORT}`)
  })

  // Connect to database after server is listening
  try {
    await prisma.$connect()
    console.log('Database connected')
  } catch (error) {
    console.error('Database connection failed:', error)
    // Don't exit — server is running, health endpoint will report unhealthy
  }
}

main().catch((error) => {
  console.error('Failed to start server:', error)
  process.exit(1)
})

export default app
