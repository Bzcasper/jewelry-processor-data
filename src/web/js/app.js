// src/web/static/js/app.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import UploadManager from './components/UploadManager.vue'
import ProcessingOptions from './components/ProcessingOptions.vue'
import ResultsViewer from './components/ResultsViewer.vue'

const app = createApp({
    components: {
        UploadManager,
        ProcessingOptions,
        ResultsViewer
    }
})

app.use(createPinia())