console.log('Starting AutoBuilder backend...')
console.log('PORT env:', process.env.PORT)
console.log('NODE_ENV:', process.env.NODE_ENV)

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

const PORT = Number(process.env.PORT) || 3001

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`AutoBuilder API running on port ${PORT}`)
})

server.on('error', (err) => {
  console.error('Server listen error:', err)
  process.exit(1)
})

// Connect to database in background (don't block server startup)
prisma.$connect()
  .then(() => console.log('Database connected'))
  .catch((err) => console.error('Database connection failed:', err))

export default app
