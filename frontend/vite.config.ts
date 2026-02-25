import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: parseInt(process.env.VITE_PORT || '3000'),
        strictPort: false,
        host: process.env.VITE_HOST || 'localhost',
        proxy: {
            '/api': {
                target: process.env.VITE_API_URL || 'http://localhost:8000',
                changeOrigin: true,
            },
            '/ws': {
                target: process.env.VITE_WS_URL || 'ws://localhost:8000',
                ws: true,
                changeOrigin: true,
            },
        },
    },
    build: {
        target: 'esnext',
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
            },
        },
        rollupOptions: {
            output: {
                // Code splitting configuration
                manualChunks: {
                    vendor: ['react', 'react-dom', 'zustand'],
                    utils: ['date-fns'],
                },
                entryFileNames: 'js/[name]-[hash].js',
                chunkFileNames: 'js/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash][extname]',
            },
        },
        // Enable CSS code splitting
        cssCodeSplit: true,
        // Reporting compressed size
        reportCompressedSize: true,
        // Chunk size warning
        chunkSizeWarningLimit: 500,
    },
    // Optimization settings
    optimizeDeps: {
        include: ['react', 'react-dom', 'zustand'],
        exclude: ['@vitest/ui'],
    },
})
