FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npx prisma generate
RUN npm run build

EXPOSE ${PORT:-3001}
CMD ["sh", "-c", "npx prisma db push --accept-data-loss && node dist/index.js"]
