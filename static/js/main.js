// DOM Elements
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('mri-file');
const previewContainer = document.getElementById('preview-container');
const resultsContainer = document.getElementById('results-container');
const loadingIndicator = document.getElementById('loading');
const gradcamContainer = document.getElementById('gradcam-container');

// Class names for classification
const classNames = [
    'NonDemented', 
    'VeryMildDemented', 
    'MildDemented', 
    'ModerateDemented'
];

// Class descriptions for displaying in the results
const classDescriptions = {
    'NonDemented': 'No signs of Alzheimer\'s disease detected in the MRI scan.',
    'VeryMildDemented': 'Very mild signs of Alzheimer\'s disease detected. Early intervention recommended.',
    'MildDemented': 'Mild signs of Alzheimer\'s disease detected. Medical consultation advised.',
    'ModerateDemented': 'Moderate signs of Alzheimer\'s disease detected. Immediate medical attention recommended.'
};

// Class to CSS class mapping for styling
const classStyles = {
    'NonDemented': 'non-demented',
    'VeryMildDemented': 'very-mild-demented',
    'MildDemented': 'mild-demented',
    'ModerateDemented': 'moderate-demented'
};

// Progress bar color classes
const progressColors = {
    'NonDemented': 'bg-success',
    'VeryMildDemented': 'bg-warning',
    'MildDemented': 'bg-danger',
    'ModerateDemented': 'bg-danger'
};

// Severity badge classes
const severityBadges = {
    'NonDemented': 'bg-success',
    'VeryMildDemented': 'bg-warning',
    'MildDemented': 'bg-danger',
    'ModerateDemented': 'bg-danger'
};

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Add event listener for form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Add event listener for file input change
    fileInput.addEventListener('change', handleFileChange);
    
    // Set initial state for the drag-drop zone
    setupDragAndDrop();
    
    // Initialize theme toggler
    initThemeToggle();
});

/**
 * Initialize theme toggle functionality
 */
function initThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;
    const storedTheme = localStorage.getItem('theme');
    
    // Set the initial theme based on localStorage or default to dark
    if (storedTheme === 'light') {
        setLightTheme();
    } else {
        setDarkTheme();
    }
    
    // Add click event listener to toggle theme
    themeToggle.addEventListener('click', () => {
        if (htmlElement.classList.contains('dark-mode')) {
            setLightTheme();
        } else {
            setDarkTheme();
        }
    });
    
    // Set light theme
    function setLightTheme() {
        htmlElement.classList.remove('dark-mode');
        htmlElement.classList.add('light-mode');
        htmlElement.setAttribute('data-bs-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
    
    // Set dark theme
    function setDarkTheme() {
        htmlElement.classList.remove('light-mode');
        htmlElement.classList.add('dark-mode');
        htmlElement.setAttribute('data-bs-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    }
}

/**
 * Setup drag and drop functionality for the upload area
 */
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    if (!uploadArea) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('highlight');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('highlight');
        });
    });
    
    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFileChange({ target: { files: files } });
        }
    });
}

/**
 * Handle form submission
 * @param {Event} event - Form submit event
 */
function handleFormSubmit(event) {
    event.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select an image file first.');
        return;
    }
    
    // Check if file is an image
    if (!file.type.match('image.*')) {
        showError('Please select a valid image file (JPG, PNG).');
        return;
    }
    
    // Show loading indicator
    loadingIndicator.classList.remove('d-none');
    resultsContainer.innerHTML = '';
    gradcamContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Generating visualization...</p>
        </div>
    `;
    
    // Create form data and submit to backend
    const formData = new FormData();
    formData.append('file', file);
    
    // Send the image to the backend for analysis with timeout
    const fetchPromise = fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    // Add a timeout to the fetch call to prevent hanging
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timed out')), 30000); // 30 seconds timeout
    });
    
    Promise.race([fetchPromise, timeoutPromise])
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Server error occurred');
            }).catch(e => {
                throw new Error('Network response was not ok: ' + response.status);
            });
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Display the results
        displayResults(data);
        
        // Display Grad-CAM visualization if available
        if (data.cam_image) {
            displayGradCAM(data.cam_image, data.original_image);
        }
    })
    .catch(error => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        gradcamContainer.innerHTML = '<p class="text-muted">Visualization unavailable</p>';
        showError('Error analyzing image: ' + error.message);
    });
}

/**
 * Handle file input change
 * @param {Event} event - File input change event
 */
function handleFileChange(event) {
    const file = event.target.files[0];
    if (!file) {
        previewContainer.innerHTML = '<p class="text-muted">No image selected</p>';
        return;
    }
    
    // Check if file is an image
    if (!file.type.match('image.*')) {
        previewContainer.innerHTML = '<p class="text-danger">Please select a valid image file (JPG, PNG).</p>';
        return;
    }
    
    // Update file name display
    const fileNameDisplay = document.getElementById('file-name');
    if (fileNameDisplay) {
        fileNameDisplay.textContent = file.name;
    }
    
    // Display image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewContainer.innerHTML = `
            <div class="position-relative">
                <img src="${e.target.result}" alt="MRI Preview" class="img-fluid rounded shadow">
                <div class="position-absolute top-0 end-0 m-2">
                    <span class="badge bg-secondary text-white px-3 py-2">Preview</span>
                </div>
            </div>
        `;
    };
    reader.readAsDataURL(file);
    
    // Clear previous results
    resultsContainer.innerHTML = '<p class="text-muted text-center">Click "Analyze Image" to see results</p>';
    gradcamContainer.innerHTML = '<p class="text-muted">Visualization will appear here after analysis</p>';
}

/**
 * Display analysis results
 * @param {Object} data - Results data from backend
 */
function displayResults(data) {
    // Get the predicted class and confidence
    const predictedClass = data.class;
    const confidence = data.confidence * 100;
    
    // Create results HTML
    let resultsHTML = `
        <div class="card border-0 shadow-sm mb-4">
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    <div class="me-3">
                        <span class="badge ${severityBadges[predictedClass]} p-3 rounded-circle">
                            <i class="bi bi-clipboard2-pulse fs-5"></i>
                        </span>
                    </div>
                    <div>
                        <h4 class="mb-0">Classification Result</h4>
                        <p class="text-muted mb-0">AI-based analysis</p>
                    </div>
                </div>
                
                <div class="bg-dark bg-opacity-10 p-3 rounded mb-3">
                    <div class="row align-items-center">
                        <div class="col-md-4">
                            <h5>${predictedClass}</h5>
                            <span class="badge ${severityBadges[predictedClass]}">${confidence.toFixed(2)}% confidence</span>
                        </div>
                        <div class="col-md-8">
                            <div class="progress" style="height: 1.5rem;">
                                <div class="progress-bar ${progressColors[predictedClass]}" 
                                     role="progressbar" 
                                     style="width: ${confidence}%" 
                                     aria-valuenow="${confidence}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                    ${confidence.toFixed(2)}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="alert alert-secondary">
                    <p class="mb-0">${classDescriptions[predictedClass] || 'No description available.'}</p>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
}

/**
 * Display Grad-CAM visualization
 * @param {string} base64Image - Base64 encoded image data for the heatmap
 * @param {string} originalImage - Base64 encoded original image data
 */
function displayGradCAM(base64Image, originalImage) {
    gradcamContainer.innerHTML = `
        <div class="card border-0 shadow-sm">
            <div class="card-body">
                <h5 class="card-title mb-3">Region Analysis</h5>
                <div class="row">
                    <div class="col-md-6 mb-3 mb-md-0">
                        <div class="position-relative">
                            <img src="data:image/jpeg;base64,${originalImage}" alt="Original MRI" class="img-fluid rounded shadow-sm">
                            <div class="position-absolute top-0 end-0 m-2">
                                <span class="badge bg-secondary text-white px-2 py-1">Original</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="position-relative">
                            <img src="data:image/jpeg;base64,${base64Image}" alt="Grad-CAM Visualization" class="img-fluid rounded shadow-sm">
                            <div class="position-absolute top-0 end-0 m-2">
                                <span class="badge bg-secondary text-white px-2 py-1">Heatmap</span>
                            </div>
                        </div>
                    </div>
                </div>
                <p class="mt-3 text-muted small">
                    <i class="bi bi-info-circle me-1"></i>
                    The heatmap highlights regions of interest that contributed to the classification decision. 
                    Red/yellow areas indicate regions of higher importance.
                </p>
            </div>
        </div>
    `;
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    resultsContainer.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <div class="d-flex">
                <div class="me-3">
                    <i class="bi bi-exclamation-triangle-fill fs-4"></i>
                </div>
                <div>
                    <h5 class="alert-heading">Error</h5>
                    <p class="mb-0">${message}</p>
                </div>
            </div>
        </div>
    `;
}