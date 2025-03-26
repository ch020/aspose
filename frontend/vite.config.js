import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'fs';

// https://vite.dev/config/
export default defineConfig({
    server: {
        host: true,
        https: {
            key: fs.readFileSync('../key.pem'),
            cert: fs.readFileSync('../cert.pem'),
        },
    },
  plugins: [
      react(),
      tailwindcss(),
  ],
})
