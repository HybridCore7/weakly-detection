/**
 * DeepFake Detection — Frontend Application Logic
 * 
 * Handles file upload, API communication, and result visualization
 * with smooth animations and real-time progress updates.
 */

// ============================================================
// DOM Elements
// ============================================================

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadIcon = document.getElementById('uploadIcon');
const uploadTitle = document.getElementById('uploadTitle');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const previewVideo = document.getElementById('previewVideo');
const previewFilename = document.getElementById('previewFilename');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const loadingStep = document.getElementById('loadingStep');
const progressBar = document.getElementById('progressBar');
const resultsPanel = document.getElementById('resultsPanel');

// Result elements
const verdictCard = document.getElementById('verdictCard');
const verdictIcon = document.getElementById('verdictIcon');
const verdictLabel = document.getElementById('verdictLabel');
const verdictText = document.getElementById('verdictText');
const gaugeFill = document.getElementById('gaugeFill');
const gaugeValue = document.getElementById('gaugeValue');
const realBar = document.getElementById('realBar');
const fakeBar = document.getElementById('fakeBar');
const realValue = document.getElementById('realValue');
const fakeValue = document.getElementById('fakeValue');
const modelList = document.getElementById('modelList');
const regionsCard = document.getElementById('regionsCard');
const regionsList = document.getElementById('regionsList');
const timelineCard = document.getElementById('timelineCard');
const timelineBar = document.getElementById('timelineBar');
const timelineInfo = document.getElementById('timelineInfo');
const analysisTime = document.getElementById('analysisTime');
const faceDetected = document.getElementById('faceDetected');
const analysisMode = document.getElementById('analysisMode');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

let selectedFile = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    setupNavigation();
});

function setupEventListeners() {
    // Upload zone click
    uploadZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear button
    clearBtn.addEventListener('click', resetUpload);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeFile);
}

function setupNavigation() {
    // Smooth scroll navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        const navbar = document.getElementById('navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 10, 15, 0.95)';
        } else {
            navbar.style.background = 'rgba(10, 10, 15, 0.8)';
        }
    });
}

// ============================================================
// Health Check
// ============================================================

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        statusIndicator.style.background = '#10b981';
        statusText.textContent = data.demo_mode ? 'Demo Mode' : 'Model Active';
    } catch (error) {
        statusIndicator.style.background = '#ef4444';
        statusText.textContent = 'Offline';
    }
}

// ============================================================
// File Handling
// ============================================================

function handleFile(file) {
    const imageExts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];
    const videoExts = ['.mp4', '.avi', '.mov', '.mkv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    const isImage = imageExts.includes(ext);
    const isVideo = videoExts.includes(ext);

    if (!isImage && !isVideo) {
        showNotification('Unsupported file format', 'error');
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        showNotification('File too large (max 100MB)', 'error');
        return;
    }

    selectedFile = file;
    previewFilename.textContent = file.name;

    // Show preview
    const url = URL.createObjectURL(file);

    if (isImage) {
        previewImage.src = url;
        previewImage.style.display = 'block';
        previewVideo.style.display = 'none';
    } else {
        previewVideo.src = url;
        previewVideo.style.display = 'block';
        previewImage.style.display = 'none';
    }

    // Show preview area, hide upload zone
    uploadZone.style.display = 'none';
    previewArea.style.display = 'block';
    resultsPanel.style.display = 'none';
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewVideo.src = '';

    uploadZone.style.display = 'flex';
    previewArea.style.display = 'none';
    loadingState.style.display = 'none';
    resultsPanel.style.display = 'none';
}

// ============================================================
// Analysis
// ============================================================

async function analyzeFile() {
    if (!selectedFile) return;

    // Show loading state
    previewArea.style.display = 'none';
    loadingState.style.display = 'flex';
    resultsPanel.style.display = 'none';

    // Animate loading steps
    const steps = [
        'Extracting faces...',
        'Running XceptionNet...',
        'Running EfficientNet...',
        'Analyzing with Autoencoder...',
        'Computing contrastive embeddings...',
        'Fusing ensemble predictions...',
        'Generating heatmaps...',
        'Finalizing analysis...',
    ];

    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        if (stepIndex < steps.length) {
            loadingStep.textContent = steps[stepIndex];
            const progress = ((stepIndex + 1) / steps.length) * 90;
            progressBar.style.width = progress + '%';
            stepIndex++;
        }
    }, 400);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        });

        clearInterval(stepInterval);
        progressBar.style.width = '100%';

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await response.json();

        // Short delay for visual completion
        await new Promise(resolve => setTimeout(resolve, 500));

        displayResults(result);

    } catch (error) {
        clearInterval(stepInterval);
        showNotification(error.message || 'Analysis failed', 'error');
        previewArea.style.display = 'block';
        loadingState.style.display = 'none';
    }
}

// ============================================================
// Results Display
// ============================================================

function displayResults(result) {
    loadingState.style.display = 'none';
    previewArea.style.display = 'block';
    resultsPanel.style.display = 'flex';

    const isFake = result.prediction === 'Fake';
    const confidence = result.confidence;

    // Verdict
    verdictCard.className = 'verdict-card glass-card ' + (isFake ? 'verdict-fake' : 'verdict-real');
    verdictIcon.className = 'verdict-icon ' + (isFake ? 'fake' : 'real');
    verdictIcon.textContent = isFake ? '⚠️' : '✅';
    verdictLabel.textContent = 'VERDICT';
    verdictText.textContent = result.prediction.toUpperCase();
    verdictText.className = 'verdict-text ' + (isFake ? 'fake' : 'real');

    // Gauge animation
    const fakeProb = result.probabilities.Fake;
    setTimeout(() => {
        gaugeFill.style.width = (fakeProb * 100) + '%';
        gaugeFill.className = 'gauge-fill ' + (isFake ? 'fake' : 'real');
        gaugeValue.textContent = (confidence * 100).toFixed(1) + '%';
    }, 100);

    // Probability bars
    setTimeout(() => {
        realBar.style.width = (result.probabilities.Real * 100) + '%';
        fakeBar.style.width = (result.probabilities.Fake * 100) + '%';
        realValue.textContent = (result.probabilities.Real * 100).toFixed(1) + '%';
        fakeValue.textContent = (result.probabilities.Fake * 100).toFixed(1) + '%';
    }, 200);

    // Model contributions
    if (result.model_contributions) {
        modelList.innerHTML = '';
        const models = result.model_contributions;

        Object.entries(models).forEach(([name, data], index) => {
            const div = document.createElement('div');
            div.className = 'model-contrib';
            div.style.animationDelay = (index * 0.1) + 's';

            const predClass = data.prediction === 'Fake' ? 'fake' : 'real';
            const barColor = data.prediction === 'Fake'
                ? 'var(--accent-danger)'
                : 'var(--accent-success)';

            div.innerHTML = `
                <span class="model-contrib-name">${name}</span>
                <div class="model-contrib-bar">
                    <div class="model-contrib-fill" style="width: 0%; background: ${barColor};"></div>
                </div>
                <span class="model-contrib-value">${(data.confidence * 100).toFixed(1)}%</span>
                <span class="model-contrib-pred ${predClass}">${data.prediction}</span>
            `;

            modelList.appendChild(div);

            // Animate bar
            setTimeout(() => {
                const fill = div.querySelector('.model-contrib-fill');
                fill.style.width = (data.confidence * 100) + '%';
            }, 300 + index * 150);
        });
    }

    // Suspicious regions
    if (result.suspicious_regions && result.suspicious_regions.length > 0) {
        regionsCard.style.display = 'block';
        regionsList.innerHTML = '';

        result.suspicious_regions.forEach(region => {
            const div = document.createElement('div');
            div.className = 'region-item';
            div.innerHTML = `
                <div>
                    <span class="region-name">${region.region.replace('_', ' ')}</span>
                    <span class="region-type">${region.type}</span>
                </div>
                <span class="region-score">${(region.anomaly_score * 100).toFixed(1)}%</span>
            `;
            regionsList.appendChild(div);
        });
    } else {
        regionsCard.style.display = 'none';
    }

    // Video timeline
    if (result.timeline && result.timeline.length > 0) {
        timelineCard.style.display = 'block';
        timelineBar.innerHTML = '';

        result.timeline.forEach(frame => {
            const div = document.createElement('div');
            div.className = 'timeline-frame ' + (frame.prediction === 'Fake' ? 'fake' : 'real');
            div.style.opacity = 0.3 + frame.fake_probability * 0.7;
            div.title = `Frame ${frame.frame_index}: ${frame.prediction} (${(frame.fake_probability * 100).toFixed(1)}%)`;
            timelineBar.appendChild(div);
        });

        const fakeFrames = result.timeline.filter(f => f.prediction === 'Fake').length;
        timelineInfo.innerHTML = `
            <span>${result.frames_analyzed} frames analyzed</span>
            <span>${fakeFrames} suspicious frames (${(result.fake_frame_ratio * 100).toFixed(1)}%)</span>
        `;
    } else {
        timelineCard.style.display = 'none';
    }

    // Meta info
    analysisTime.textContent = result.analysis_time ? result.analysis_time + 's' : '—';
    faceDetected.textContent = result.face_detected ? 'Yes ✓' : 'No ✗';
    analysisMode.textContent = result.demo_mode ? 'Demo' : 'Production';

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// Notifications
// ============================================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 24px;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        font-family: 'Inter', sans-serif;
        max-width: 360px;
    `;

    const colors = {
        info: { bg: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', border: 'rgba(99, 102, 241, 0.3)' },
        success: { bg: 'rgba(16, 185, 129, 0.15)', color: '#34d399', border: 'rgba(16, 185, 129, 0.3)' },
        error: { bg: 'rgba(239, 68, 68, 0.15)', color: '#f87171', border: 'rgba(239, 68, 68, 0.3)' },
    };

    const c = colors[type] || colors.info;
    notification.style.background = c.bg;
    notification.style.color = c.color;
    notification.style.borderColor = c.border;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(20px)';
        notification.style.transition = '0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}
