// Initialize charts
let latencyChart, modelChart;

document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    loadStats();
    
    // Refresh stats every 30 seconds
    setInterval(loadStats, 30000);
});

function initCharts() {
    // Latency Chart - FIXED (no more growing!)
    const latencyCtx = document.getElementById('latencyChart').getContext('2d');
    latencyChart = new Chart(latencyCtx, {
        type: 'line',
        data: {
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
            datasets: [{
                label: 'Response Time (ms)',
                data: [25, 28, 22, 30, 27, 24, 26],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,  // ✅ CHANGED: false (not true!)
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 40,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#cbd5e1'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#cbd5e1'
                    }
                }
            }
        }
    });

    // Model Performance Chart - ALSO FIXED
    const modelCtx = document.getElementById('modelChart').getContext('2d');
    modelChart = new Chart(modelCtx, {
        type: 'bar',
        data: {
            labels: ['Collaborative', 'Content-Based', 'Neural Net', 'Hybrid'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [89.5, 87.2, 91.8, 92.3],
                backgroundColor: [
                    'rgba(168, 85, 247, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(34, 197, 94, 0.7)',
                    'rgba(249, 115, 22, 0.7)'
                ],
                borderColor: [
                    '#a855f7',
                    '#3b82f6',
                    '#22c55e',
                    '#f97316'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,  // ✅ CHANGED: false (not true!)
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#cbd5e1'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#cbd5e1'
                    }
                }
            }
        }
    });
}

async function loadStats() {
    try {
        const response = await axios.get('/api/v1/stats');
        console.log('Stats loaded:', response.data);
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function getRecommendations() {
    const userId = document.getElementById('userId').value;
    const numRecs = document.getElementById('numRecs').value;
    
    if (!userId) {
        alert('Please enter a user ID');
        return;
    }
    
    try {
        const startTime = performance.now();
        
        const response = await axios.get(`/api/v1/recommend/${userId}?n=${numRecs}`);
        
        const endTime = performance.now();
        const latency = (endTime - startTime).toFixed(2);
        
        displayRecommendations(response.data, latency);
        
    } catch (error) {
        console.error('Error getting recommendations:', error);
        alert('Error getting recommendations. Please try again.');
    }
}

function displayRecommendations(data, clientLatency) {
    const container = document.getElementById('resultsContainer');
    const recsGrid = document.getElementById('recommendations');
    const latencyEl = document.getElementById('latency');
    const modelEl = document.getElementById('modelUsed');
    
    // Clear previous results
    recsGrid.innerHTML = '';
    
    // Display recommendations
    data.recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.innerHTML = `
            <h4>#${index + 1} ${rec.title}</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0;">
                ${rec.genres || 'Various genres'}
            </p>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                <span style="color: #10b981; font-weight: 700;">
                    ⭐ ${rec.score || 'N/A'}
                </span>
                <span style="color: #a855f7; font-size: 0.85rem;">
                    ${rec.method || 'hybrid'}
                </span>
            </div>
        `;
        recsGrid.appendChild(card);
    });
    
    // Update metadata
    latencyEl.textContent = data.latency_ms || clientLatency;
    modelEl.textContent = 'Hybrid v1';
    
    // Show results
    container.style.display = 'block';
    
    // Smooth scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}