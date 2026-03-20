-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "App" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "frontendCode" TEXT NOT NULL,
    "backendCode" TEXT NOT NULL,
    "databaseSchema" TEXT NOT NULL,
    "deploymentUrl" TEXT,
    "deploymentStatus" TEXT NOT NULL DEFAULT 'draft',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "App_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Analytics" (
    "id" TEXT NOT NULL,
    "appId" TEXT NOT NULL,
    "actionType" TEXT NOT NULL,
    "actionData" JSONB,
    "errorMessage" TEXT,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Analytics_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MetaState" (
    "id" TEXT NOT NULL,
    "appId" TEXT NOT NULL,
    "predictionError" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "complexity" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "entropy" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "novelty" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "phi" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "opacity" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "bindingStrength" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "taskDifficulty" DOUBLE PRECISION NOT NULL DEFAULT 0.3,
    "distributionShift" DOUBLE PRECISION NOT NULL DEFAULT 0.1,
    "anomalyScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MetaState_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Improvement" (
    "id" TEXT NOT NULL,
    "appId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'suggested',
    "riskLevel" TEXT NOT NULL DEFAULT 'low',
    "appliedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Improvement_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE INDEX "User_email_idx" ON "User"("email");

-- CreateIndex
CREATE INDEX "App_userId_idx" ON "App"("userId");

-- CreateIndex
CREATE INDEX "Analytics_appId_idx" ON "Analytics"("appId");

-- CreateIndex
CREATE INDEX "Analytics_timestamp_idx" ON "Analytics"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "MetaState_appId_key" ON "MetaState"("appId");

-- CreateIndex
CREATE INDEX "Improvement_appId_idx" ON "Improvement"("appId");

-- CreateIndex
CREATE INDEX "Improvement_status_idx" ON "Improvement"("status");

-- AddForeignKey
ALTER TABLE "App" ADD CONSTRAINT "App_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Analytics" ADD CONSTRAINT "Analytics_appId_fkey" FOREIGN KEY ("appId") REFERENCES "App"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MetaState" ADD CONSTRAINT "MetaState_appId_fkey" FOREIGN KEY ("appId") REFERENCES "App"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Improvement" ADD CONSTRAINT "Improvement_appId_fkey" FOREIGN KEY ("appId") REFERENCES "App"("id") ON DELETE CASCADE ON UPDATE CASCADE;

