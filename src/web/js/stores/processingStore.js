// src/web/static/js/stores/processingStore.js
import { defineStore } from 'pinia'

export const useProcessingStore = defineStore('processing', {
    state: () => ({
        files: [],
        options: {
            enhancement: {
                denoise: true,
                denoiseStrength: 0.5,
                sharpen: true,
                sharpenStrength: 0.3,
                colorEnhancement: true,
                colorStrength: 0.5,
                detailEnhancement: true,
                detailStrength: 0.4,
                highlightRecovery: true,
                shadowRecovery: true
            },
            output: {
                format: 'jpg',
                quality: 90,
                maxSize: 2048
            },
            detection: {
                drawBoxes: true,
                minConfidence: 0.7,
                labelPosition: 'top'
            }
        },
        processing: false,
        progress: 0,
        results: []
    }),
    
    actions: {
        async processFiles() {
            this.processing = true
            this.progress = 0
            
            try {
                const formData = new FormData()
                this.files.forEach(file => formData.append('files[]', file))
                formData.append('options', JSON.stringify(this.options))
                
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                })
                
                const result = await response.json()
                this.results = result.processed_images
                
            } catch (error) {
                console.error('Processing failed:', error)
            } finally {
                this.processing = false
            }
        },
        
        updateProgress(progress) {
            this.progress = progress
        },
        
        clearResults() {
            this.results = []
        }
    }
})