/**
 * MBTA Live Transit Dashboard with AI Chatbot + Location & Map
 * Complete working version with location-based features
 */

// ========================================
// State Management
// ========================================

const state = {
    currentTab: 'overview',
    routes: { subway: [], bus: [], commuter_rail: [], ferry: [] },
    alerts: [],
    stations: [],
    refreshInterval: null,
    chatHistory: [],
    // Location state
    userLocation: null,
    nearestStation: null,
    map: null,
    userMarker: null,
    stationMarkers: [],
    mapInitialized: false
};

// Line colors for map markers
const lineColors = {
    'Red': '#DA291C',
    'Orange': '#ED8B00',
    'Blue': '#003DA5',
    'Green-B': '#00843D',
    'Green-C': '#00843D',
    'Green-D': '#00843D',
    'Green-E': '#00843D',
    'Mattapan': '#DA291C'
};

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    // Load initial data
    await Promise.all([
        loadRoutes(),
        loadDashboard(),
        loadAllAlerts()
    ]);
    
    // Setup search
    setupStationSearch();
    setupBusSearch();
    setupChatInput();
    
    // Check for saved location
    checkSavedLocation();
    
    // Start auto-refresh (every 30 seconds)
    state.refreshInterval = setInterval(refreshAll, 30000);
    
    // Update time display
    updateLastUpdateTime();
}

// ========================================
// Location Functions
// ========================================

async function checkSavedLocation() {
    try {
        const response = await fetch('/api/get-location');
        const data = await response.json();
        
        if (data.station) {
            state.userLocation = { lat: data.lat, lon: data.lon };
            state.nearestStation = {
                name: data.station,
                id: data.stop_id,
                distance: data.distance,
                lines: data.lines
            };
            updateLocationUI();
        }
    } catch (error) {
        console.log('No saved location found');
    }
}

function requestLocation() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by your browser');
        return;
    }
    
    // Update button state
    const locationBtn = document.getElementById('location-btn');
    if (locationBtn) {
        locationBtn.classList.add('loading');
    }
    
    // Show banner with loading state
    const banner = document.getElementById('location-banner');
    const bannerText = document.getElementById('location-text');
    if (banner && bannerText) {
        banner.style.display = 'flex';
        bannerText.textContent = 'Detecting your location...';
    }
    
    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            
            state.userLocation = { lat, lon };
            
            try {
                const response = await fetch('/api/set-location', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lat, lon })
                });
                
                const data = await response.json();
                
                if (data.success && data.nearest_station) {
                    state.nearestStation = data.nearest_station;
                    updateLocationUI();
                    
                    // If on map tab, update map
                    if (state.currentTab === 'map' && state.mapInitialized) {
                        updateUserMarker(lat, lon);
                        state.map.setView([lat, lon], 15);
                    }
                    
                    // Add chat message about location
                    if (state.currentTab === 'chat') {
                        addChatMessage(`📍 Got it! You're near ${data.nearest_station.name} (${data.nearest_station.distance}m away). Now you can just ask "Next train to [destination]" and I'll know where you are!`, 'bot');
                    }
                } else {
                    if (bannerText) {
                        bannerText.textContent = 'No MBTA stations found nearby';
                    }
                    setTimeout(hideLocationBanner, 3000);
                }
            } catch (error) {
                console.error('Location API error:', error);
                if (bannerText) {
                    bannerText.textContent = 'Error finding nearby stations';
                }
            }
            
            if (locationBtn) {
                locationBtn.classList.remove('loading');
            }
        },
        (error) => {
            console.error('Geolocation error:', error);
            if (locationBtn) {
                locationBtn.classList.remove('loading');
            }
            
            let message = 'Unable to get location';
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    message = 'Location permission denied. Please allow location access.';
                    break;
                case error.POSITION_UNAVAILABLE:
                    message = 'Location unavailable';
                    break;
                case error.TIMEOUT:
                    message = 'Location request timed out';
                    break;
            }
            
            if (bannerText) {
                bannerText.textContent = message;
            }
            setTimeout(hideLocationBanner, 3000);
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        }
    );
}

function updateLocationUI() {
    const banner = document.getElementById('location-banner');
    const bannerText = document.getElementById('location-text');
    const locationBtn = document.getElementById('location-btn');
    const chatStatus = document.getElementById('chat-location-status');
    const statusText = document.getElementById('status-text');
    
    if (state.nearestStation) {
        if (banner && bannerText) {
            banner.style.display = 'flex';
            bannerText.textContent = `📍 You're near ${state.nearestStation.name} (${state.nearestStation.distance}m away)`;
        }
        
        if (locationBtn) {
            locationBtn.classList.add('active');
            locationBtn.title = `Near ${state.nearestStation.name}`;
        }
        
        if (chatStatus) {
            chatStatus.textContent = `📍 Near ${state.nearestStation.name} • Ask anything about transit`;
        }
        
        if (statusText) {
            statusText.textContent = `Near ${state.nearestStation.name}`;
        }
    }
}

function hideLocationBanner() {
    const banner = document.getElementById('location-banner');
    if (banner) {
        banner.style.display = 'none';
    }
}

// ========================================
// Map Functions
// ========================================

async function initMap() {
    if (state.mapInitialized) return;
    
    const mapContainer = document.getElementById('map');
    if (!mapContainer) return;
    
    // Initialize Leaflet map centered on Boston
    state.map = L.map('map').setView([42.3601, -71.0589], 12);
    
    // Add dark-themed tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(state.map);
    
    // Load and display stations
    await loadStationsOnMap();
    
    // Add user marker if location is known
    if (state.userLocation) {
        updateUserMarker(state.userLocation.lat, state.userLocation.lon);
    }
    
    state.mapInitialized = true;
    
    // Fix map size on tab switch
    setTimeout(() => {
        state.map.invalidateSize();
    }, 100);
}

async function loadStationsOnMap() {
    try {
        const response = await fetch('/api/stations');
        const stations = await response.json();
        
        stations.forEach(station => {
            const primaryLine = station.lines[0];
            const color = lineColors[primaryLine] || '#888888';
            
            // Create custom icon
            const icon = L.divIcon({
                className: 'station-marker',
                html: `<div class="station-dot" style="background-color: ${color};"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });
            
            const marker = L.marker([station.lat, station.lon], { icon })
                .addTo(state.map)
                .bindPopup(`
                    <div class="station-popup">
                        <strong>${station.name}</strong><br>
                        <span style="color: ${color}">${station.lines.join(', ')}</span><br>
                        <button onclick="openStationModal('${station.id}')" class="popup-btn">View Arrivals</button>
                    </div>
                `);
            
            state.stationMarkers.push(marker);
        });
    } catch (error) {
        console.error('Error loading stations:', error);
    }
}

function updateUserMarker(lat, lon) {
    if (!state.map) return;
    
    // Remove existing user marker
    if (state.userMarker) {
        state.map.removeLayer(state.userMarker);
    }
    
    // Create pulsing user marker
    const userIcon = L.divIcon({
        className: 'user-marker',
        html: `<div class="user-dot"><div class="user-pulse"></div></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });
    
    state.userMarker = L.marker([lat, lon], { icon: userIcon })
        .addTo(state.map)
        .bindPopup(`<strong>📍 You are here</strong><br>${state.nearestStation ? `Near ${state.nearestStation.name}` : ''}`);
}

// ========================================
// Chat Functions
// ========================================

function setupChatInput() {
    const input = document.getElementById('chat-input');
    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
}

function sendSuggestion(text) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = text;
        sendChatMessage();
    }
}

// ==================== VOICE INPUT ====================
let recognition = null;
let isListening = false;

function initVoiceInput() {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        console.log('Speech recognition not supported');
        const micBtn = document.getElementById('chat-mic-btn');
        if (micBtn) {
            micBtn.style.display = 'none';
        }
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isListening = true;
        updateVoiceUI(true);
        console.log('Voice recognition started');
    };
    
    recognition.onresult = function(event) {
        const input = document.getElementById('chat-input');
        if (!input) return;
        
        let finalTranscript = '';
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Show interim results in input
        if (interimTranscript) {
            input.value = interimTranscript;
            input.placeholder = 'Listening...';
        }
        
        // When we have final result
        if (finalTranscript) {
            input.value = finalTranscript;
            input.placeholder = 'Ask about trains, buses, delays...';
            
            // Auto-send after getting final result
            setTimeout(() => {
                if (input.value.trim()) {
                    sendChatMessage();
                }
            }, 500);
        }
    };
    
    recognition.onerror = function(event) {
        console.log('Voice recognition error:', event.error);
        isListening = false;
        updateVoiceUI(false);
        
        const voiceStatusText = document.getElementById('voice-status-text');
        if (voiceStatusText) {
            if (event.error === 'not-allowed') {
                voiceStatusText.textContent = 'Microphone access denied. Please allow microphone access.';
            } else if (event.error === 'no-speech') {
                voiceStatusText.textContent = 'No speech detected. Try again.';
            } else {
                voiceStatusText.textContent = 'Error: ' + event.error;
            }
        }
        
        // Hide status after 3 seconds
        setTimeout(() => {
            const voiceStatus = document.getElementById('voice-status');
            if (voiceStatus) voiceStatus.style.display = 'none';
        }, 3000);
    };
    
    recognition.onend = function() {
        isListening = false;
        updateVoiceUI(false);
        console.log('Voice recognition ended');
    };
}

function toggleVoiceInput() {
    if (!recognition) {
        initVoiceInput();
        if (!recognition) {
            alert('Sorry, your browser does not support voice input. Please use Chrome, Edge, or Safari.');
            return;
        }
    }
    
    if (isListening) {
        recognition.stop();
        isListening = false;
        updateVoiceUI(false);
    } else {
        try {
            recognition.start();
        } catch (e) {
            console.log('Recognition already started or error:', e);
        }
    }
}

function updateVoiceUI(listening) {
    const micBtn = document.getElementById('chat-mic-btn');
    const voiceStatus = document.getElementById('voice-status');
    const voiceStatusText = document.getElementById('voice-status-text');
    const input = document.getElementById('chat-input');
    
    if (micBtn) {
        if (listening) {
            micBtn.classList.add('listening');
            micBtn.title = 'Stop listening';
        } else {
            micBtn.classList.remove('listening');
            micBtn.title = 'Voice input';
        }
    }
    
    if (voiceStatus) {
        voiceStatus.style.display = listening ? 'flex' : 'none';
    }
    
    if (voiceStatusText) {
        voiceStatusText.textContent = listening ? 'Listening... Speak now' : '';
    }
    
    if (input) {
        input.placeholder = listening ? 'Listening...' : 'Ask about trains, buses, delays...';
    }
}

// Initialize voice input on page load
document.addEventListener('DOMContentLoaded', function() {
    initVoiceInput();
});

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Check if location is requested
        if (data.request_location) {
            addChatMessage(data.reply, 'bot');
            // Auto-request location after a short delay
            setTimeout(() => {
                requestLocation();
            }, 1500);
        } else {
            // Add bot response
            addChatMessage(data.reply, 'bot');
        }
        
    } catch (error) {
        hideTypingIndicator();
        addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Chat error:', error);
    } finally {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
    }
}

function addChatMessage(text, sender) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    
    const avatar = sender === 'bot' ? '🚇' : '👤';
    
    // Parse markdown-like formatting
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/• /g, '<br>• ');
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${formattedText}</div>
        </div>
    `;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    state.chatHistory.push({ sender, text });
}

function showTypingIndicator() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🚇</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
}

function hideTypingIndicator() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();
}

// ========================================
// Tab Navigation
// ========================================

function switchTab(tabName) {
    state.currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-content`);
    });
    
    // Load tab-specific data
    switch(tabName) {
        case 'bus':
            loadBusRoutes();
            break;
        case 'commuter':
            loadCommuterRail();
            break;
        case 'ferry':
            loadFerryRoutes();
            break;
        case 'chat':
            const chatInput = document.getElementById('chat-input');
            if (chatInput) chatInput.focus();
            break;
        case 'map':
            // Initialize map if not already done
            if (!state.mapInitialized) {
                setTimeout(() => {
                    initMap();
                }, 100);
            } else if (state.map) {
                // Fix map display issues
                setTimeout(() => {
                    state.map.invalidateSize();
                }, 100);
            }
            break;
    }
}

// ========================================
// API Functions
// ========================================

async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`/api${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return null;
    }
}

// ========================================
// Dashboard Functions
// ========================================

async function loadDashboard() {
    const data = await fetchAPI('/dashboard');
    if (!data) return;
    
    renderLineStatusGrid(data.subway_status);
    renderOverviewAlerts(data.severe_alerts);
}

function renderLineStatusGrid(statusData) {
    const grid = document.getElementById('line-status-grid');
    if (!grid) return;
    
    if (!statusData) {
        grid.innerHTML = '<div class="empty-state">Unable to load status</div>';
        return;
    }
    
    const lines = ['Red', 'Orange', 'Blue', 'Green-B', 'Green-C', 'Green-D', 'Green-E', 'Mattapan'];
    const lineNames = {
        'Red': 'Red Line',
        'Orange': 'Orange Line',
        'Blue': 'Blue Line',
        'Green-B': 'Green B',
        'Green-C': 'Green C',
        'Green-D': 'Green D',
        'Green-E': 'Green E',
        'Mattapan': 'Mattapan'
    };
    
    grid.innerHTML = lines.map(line => {
        const status = statusData[line] || { vehicle_count: 0, alert_count: 0, status: 'normal' };
        return `
            <div class="line-status-card status-${status.status}" onclick="loadLineDetails('${line}')">
                <div class="line-name">
                    <span class="line-dot" style="background: ${lineColors[line]}"></span>
                    ${lineNames[line]}
                </div>
                <div class="line-stats">
                    <span class="stat">🚇 ${status.vehicle_count}</span>
                    <span class="stat">⚠️ ${status.alert_count}</span>
                </div>
            </div>
        `;
    }).join('');
}

function renderOverviewAlerts(alerts) {
    const container = document.getElementById('overview-alerts');
    if (!container) return;
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No severe alerts at this time</div>';
        return;
    }
    
    container.innerHTML = alerts.slice(0, 5).map(alert => renderAlertItem(alert)).join('');
}

// ========================================
// Routes Functions
// ========================================

async function loadRoutes() {
    const data = await fetchAPI('/routes');
    if (!data) return;
    
    state.routes = data;
}

async function loadBusRoutes() {
    const container = document.getElementById('bus-routes');
    if (!container) return;
    
    if (state.routes.bus.length === 0) {
        await loadRoutes();
    }
    
    if (state.routes.bus.length === 0) {
        container.innerHTML = '<div class="empty-state">Unable to load bus routes</div>';
        return;
    }
    
    container.innerHTML = state.routes.bus.map(route => `
        <button class="bus-route-btn" onclick="loadBusDetails('${route.id}')">
            <div class="route-number">${route.short_name || route.id}</div>
            <div class="route-name">${route.name}</div>
        </button>
    `).join('');
}

async function loadCommuterRail() {
    const container = document.getElementById('commuter-lines');
    if (!container) return;
    
    if (state.routes.commuter_rail.length === 0) {
        await loadRoutes();
    }
    
    if (state.routes.commuter_rail.length === 0) {
        container.innerHTML = '<div class="empty-state">Unable to load commuter rail</div>';
        return;
    }
    
    container.innerHTML = state.routes.commuter_rail.map(route => `
        <button class="line-btn commuter" onclick="loadCommuterDetails('${route.id}')">
            <span class="line-name">${route.name}</span>
            <span class="line-dest">${route.direction_destinations.join(' ↔ ')}</span>
        </button>
    `).join('');
}

async function loadFerryRoutes() {
    const container = document.getElementById('ferry-routes');
    if (!container) return;
    
    if (state.routes.ferry.length === 0) {
        await loadRoutes();
    }
    
    if (state.routes.ferry.length === 0) {
        container.innerHTML = '<div class="empty-state">No ferry routes available</div>';
        return;
    }
    
    container.innerHTML = state.routes.ferry.map(route => `
        <button class="line-btn ferry-btn" onclick="loadFerryDetails('${route.id}')">
            <span class="line-name">${route.name}</span>
            <span class="line-dest">${route.direction_destinations.join(' ↔ ')}</span>
        </button>
    `).join('');
}

// ========================================
// Line Details Functions
// ========================================

async function loadLineDetails(lineId) {
    const section = document.getElementById('line-details-section');
    const title = document.getElementById('line-details-title');
    const content = document.getElementById('line-details-content');
    
    if (!section || !title || !content) return;
    
    section.style.display = 'block';
    title.textContent = `${lineId} Line Details`;
    content.innerHTML = '<div class="loading-skeleton">Loading...</div>';
    
    const data = await fetchAPI(`/line/${lineId}`);
    if (!data) {
        content.innerHTML = '<div class="empty-state">Unable to load line details</div>';
        return;
    }
    
    renderLineDetails(data, content);
    section.scrollIntoView({ behavior: 'smooth' });
}

function renderLineDetails(data, container) {
    const { line, predictions_by_stop, vehicles, alerts } = data;
    
    let html = `
        <div class="line-info">
            <div class="info-card">
                <div class="label">Active Vehicles</div>
                <div class="value" style="color: ${line.color}">${vehicles.length}</div>
            </div>
            <div class="info-card">
                <div class="label">Service Alerts</div>
                <div class="value" style="color: ${alerts.length > 0 ? 'var(--warning)' : 'var(--success)'}">${alerts.length}</div>
            </div>
        </div>
    `;
    
    // Alerts section
    if (alerts.length > 0) {
        html += `
            <h3 style="margin: var(--spacing-md) 0; font-size: 0.9rem; color: var(--warning);">⚠️ Active Alerts</h3>
            <div class="alerts-container" style="margin-bottom: var(--spacing-lg);">
                ${alerts.map(alert => `
                    <div class="alert-item severity-${alert.severity >= 7 ? 'high' : 'medium'}">
                        <div class="alert-title">${alert.header || alert.short_header}</div>
                        ${alert.description ? `<div class="alert-description">${alert.description}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    // Predictions by stop
    if (predictions_by_stop.length > 0) {
        html += `<h3 style="margin: var(--spacing-md) 0; font-size: 0.9rem;">Upcoming Arrivals</h3>`;
        html += '<div class="predictions-list">';
        
        predictions_by_stop.slice(0, 10).forEach(stop => {
            if (stop.predictions.length > 0) {
                const pred = stop.predictions[0];
                html += `
                    <div class="prediction-item" onclick="openStationModal('${stop.stop_id}')">
                        <div class="prediction-info">
                            <div class="prediction-destination">${pred.headsign || 'Unknown'}</div>
                            <div class="prediction-stop">${stop.stop_name}</div>
                        </div>
                        <div class="prediction-time">
                            <div class="time-away ${pred.time_away === 'Arriving' ? 'arriving' : ''}">${pred.time_away}</div>
                            <div class="time-scheduled">${pred.time_formatted || ''}</div>
                        </div>
                    </div>
                `;
            }
        });
        
        html += '</div>';
    }
    
    container.innerHTML = html;
}

function closeLineDetails() {
    const section = document.getElementById('line-details-section');
    if (section) section.style.display = 'none';
}

// ========================================
// Bus Details Functions
// ========================================

async function loadBusDetails(routeId) {
    const section = document.getElementById('bus-details-section');
    const title = document.getElementById('bus-details-title');
    const content = document.getElementById('bus-details-content');
    
    if (!section || !title || !content) return;
    
    section.style.display = 'block';
    title.textContent = `Route ${routeId} Details`;
    content.innerHTML = '<div class="loading-skeleton">Loading...</div>';
    
    const data = await fetchAPI(`/line/${routeId}`);
    if (!data) {
        content.innerHTML = '<div class="empty-state">Unable to load bus details</div>';
        return;
    }
    
    renderLineDetails(data, content);
    section.scrollIntoView({ behavior: 'smooth' });
}

function closeBusDetails() {
    const section = document.getElementById('bus-details-section');
    if (section) section.style.display = 'none';
}

// ========================================
// Commuter Rail Details Functions
// ========================================

async function loadCommuterDetails(routeId) {
    const section = document.getElementById('commuter-details-section');
    const title = document.getElementById('commuter-details-title');
    const content = document.getElementById('commuter-details-content');
    
    if (!section || !title || !content) return;
    
    section.style.display = 'block';
    title.textContent = routeId.replace('CR-', '') + ' Line';
    content.innerHTML = '<div class="loading-skeleton">Loading...</div>';
    
    const data = await fetchAPI(`/line/${routeId}`);
    if (!data) {
        content.innerHTML = '<div class="empty-state">Unable to load details</div>';
        return;
    }
    
    renderLineDetails(data, content);
    section.scrollIntoView({ behavior: 'smooth' });
}

function closeCommuterDetails() {
    const section = document.getElementById('commuter-details-section');
    if (section) section.style.display = 'none';
}

// ========================================
// Ferry Details Functions
// ========================================

async function loadFerryDetails(routeId) {
    const section = document.getElementById('ferry-details-section');
    const title = document.getElementById('ferry-details-title');
    const content = document.getElementById('ferry-details-content');
    
    if (!section || !title || !content) return;
    
    section.style.display = 'block';
    title.textContent = 'Ferry Details';
    content.innerHTML = '<div class="loading-skeleton">Loading...</div>';
    
    const data = await fetchAPI(`/line/${routeId}`);
    if (!data) {
        content.innerHTML = '<div class="empty-state">Unable to load ferry details</div>';
        return;
    }
    
    renderLineDetails(data, content);
    section.scrollIntoView({ behavior: 'smooth' });
}

function closeFerryDetails() {
    const section = document.getElementById('ferry-details-section');
    if (section) section.style.display = 'none';
}

// ========================================
// Alerts Functions
// ========================================

async function loadAllAlerts() {
    const data = await fetchAPI('/alerts');
    if (!data) return;
    
    state.alerts = data.alerts;
    const alertCount = document.getElementById('alert-count');
    if (alertCount) alertCount.textContent = data.count;
    
    renderAllAlerts(state.alerts);
}

function renderAllAlerts(alerts) {
    const container = document.getElementById('all-alerts');
    if (!container) return;
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No active alerts</div>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => renderAlertItem(alert)).join('');
}

function renderAlertItem(alert) {
    const severityClass = alert.severity >= 7 ? 'high' : (alert.severity >= 4 ? 'medium' : 'low');
    const routeTags = (alert.affected_routes || []).slice(0, 3).map(route => {
        const routeClass = getRouteClass(route);
        return `<span class="route-tag ${routeClass}">${route}</span>`;
    }).join('');
    
    return `
        <div class="alert-item severity-${severityClass}" onclick="toggleAlertExpand(this)">
            <div class="alert-header">
                <div class="alert-routes">${routeTags}</div>
                <span class="alert-effect">${alert.effect || alert.effect_name || ''}</span>
            </div>
            <div class="alert-title">${alert.header || alert.short_header || 'Service Alert'}</div>
            <div class="alert-description">${alert.description || ''}</div>
        </div>
    `;
}

function getRouteClass(route) {
    if (route.startsWith('Red') || route === 'Mattapan') return 'red';
    if (route.startsWith('Orange')) return 'orange';
    if (route.startsWith('Blue')) return 'blue';
    if (route.startsWith('Green')) return 'green';
    if (route.startsWith('CR-')) return 'commuter';
    if (route.startsWith('Boat')) return 'ferry';
    if (route.match(/^\d+$/)) return 'bus';
    return '';
}

function toggleAlertExpand(element) {
    element.classList.toggle('expanded');
}

function filterAlerts(type) {
    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase() === type);
    });
    
    let filtered = state.alerts;
    
    if (type !== 'all') {
        filtered = state.alerts.filter(alert => {
            const routes = alert.affected_routes || [];
            switch(type) {
                case 'subway':
                    return routes.some(r => r.startsWith('Red') || r.startsWith('Orange') || r.startsWith('Blue') || r.startsWith('Green'));
                case 'bus':
                    return routes.some(r => r.match(/^\d+$/));
                case 'commuter':
                    return routes.some(r => r.startsWith('CR-'));
                case 'ferry':
                    return routes.some(r => r.startsWith('Boat'));
                default:
                    return true;
            }
        });
    }
    
    renderAllAlerts(filtered);
}

// ========================================
// Station Search Functions
// ========================================

function setupStationSearch() {
    const input = document.getElementById('station-search');
    const results = document.getElementById('search-results');
    
    if (!input || !results) return;
    
    input.addEventListener('input', async (e) => {
        const query = e.target.value.trim();
        
        if (query.length < 2) {
            results.classList.remove('active');
            return;
        }
        
        // Search stops
        const data = await fetchAPI('/stops?route=Red,Orange,Blue,Green-B,Green-C,Green-D,Green-E');
        if (!data) return;
        
        const filtered = data.stops.filter(stop => 
            stop.name.toLowerCase().includes(query.toLowerCase())
        ).slice(0, 10);
        
        if (filtered.length > 0) {
            results.innerHTML = filtered.map(stop => `
                <div class="search-result-item" onclick="openStationModal('${stop.id}')">
                    <strong>${stop.name}</strong>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">${stop.municipality || ''}</div>
                </div>
            `).join('');
            results.classList.add('active');
        } else {
            results.classList.remove('active');
        }
    });
    
    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box')) {
            results.classList.remove('active');
        }
    });
}

function setupBusSearch() {
    const input = document.getElementById('bus-search');
    if (!input) return;
    
    input.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        const buttons = document.querySelectorAll('.bus-route-btn');
        
        buttons.forEach(btn => {
            const routeNum = btn.querySelector('.route-number');
            const routeName = btn.querySelector('.route-name');
            const numText = routeNum ? routeNum.textContent.toLowerCase() : '';
            const nameText = routeName ? routeName.textContent.toLowerCase() : '';
            const match = numText.includes(query) || nameText.includes(query);
            btn.style.display = match ? '' : 'none';
        });
    });
}

// ========================================
// Station Modal Functions
// ========================================

async function openStationModal(stopId) {
    const modal = document.getElementById('station-modal');
    const title = document.getElementById('modal-station-name');
    const body = document.getElementById('modal-body');
    
    if (!modal || !title || !body) return;
    
    modal.classList.add('active');
    title.textContent = 'Loading...';
    body.innerHTML = '<div class="loading-skeleton">Loading station data...</div>';
    
    const data = await fetchAPI(`/station/${stopId}`);
    if (!data) {
        body.innerHTML = '<div class="empty-state">Unable to load station data</div>';
        return;
    }
    
    title.textContent = data.station.name;
    
    let html = '';
    
    // Predictions
    if (data.predictions.length > 0) {
        html += `<h3 style="font-size: 0.9rem; margin-bottom: var(--spacing-sm);">Upcoming Arrivals</h3>`;
        html += '<div class="predictions-list">';
        
        data.predictions.forEach(pred => {
            html += `
                <div class="prediction-item">
                    <div class="prediction-info">
                        <div class="prediction-destination" style="color: ${pred.route_color}">${pred.headsign || pred.route_name}</div>
                        <div class="prediction-stop">${pred.route_name}</div>
                    </div>
                    <div class="prediction-time">
                        <div class="time-away ${pred.time_away === 'Arriving' ? 'arriving' : ''}">${pred.time_away}</div>
                        <div class="time-scheduled">${pred.arrival_formatted || ''}</div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    } else {
        html += '<div class="empty-state">No upcoming arrivals</div>';
    }
    
    body.innerHTML = html;
}

function closeModal() {
    const modal = document.getElementById('station-modal');
    if (modal) modal.classList.remove('active');
}

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// ========================================
// Refresh Functions
// ========================================

async function refreshAll() {
    updateLastUpdateTime();
    
    await Promise.all([
        loadDashboard(),
        loadAllAlerts()
    ]);
    
    // Refresh current tab if needed
    switch(state.currentTab) {
        case 'bus':
            loadBusRoutes();
            break;
        case 'commuter':
            loadCommuterRail();
            break;
        case 'ferry':
            loadFerryRoutes();
            break;
    }
}

function updateLastUpdateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
    });
    const el = document.getElementById('last-update-time');
    if (el) el.textContent = timeStr;
}

// ==================== TRIP PLANNER ====================
let allStations = [];
let selectedOrigin = null;
let selectedDestination = null;

// Load stations for autocomplete
async function loadStationsForTrip() {
    try {
        const response = await fetch('/api/stations');
        allStations = await response.json();
        console.log(`Loaded ${allStations.length} stations for trip planner`);
    } catch (error) {
        console.error('Failed to load stations:', error);
    }
}

// Setup trip planner event listeners
function setupTripPlanner() {
    const originInput = document.getElementById('trip-origin');
    const destInput = document.getElementById('trip-destination');
    
    if (originInput) {
        originInput.addEventListener('input', (e) => handleTripAutocomplete(e.target, 'origin'));
        originInput.addEventListener('focus', (e) => handleTripAutocomplete(e.target, 'origin'));
        originInput.addEventListener('blur', () => setTimeout(() => hideAutocomplete('origin'), 200));
    }
    
    if (destInput) {
        destInput.addEventListener('input', (e) => handleTripAutocomplete(e.target, 'dest'));
        destInput.addEventListener('focus', (e) => handleTripAutocomplete(e.target, 'dest'));
        destInput.addEventListener('blur', () => setTimeout(() => hideAutocomplete('dest'), 200));
    }
    
    // Load stations
    loadStationsForTrip();
}

function handleTripAutocomplete(input, type) {
    const query = input.value.toLowerCase().trim();
    const container = document.getElementById(`${type}-autocomplete`);
    
    if (!container) return;
    
    if (query.length < 1) {
        container.classList.remove('active');
        return;
    }
    
    const matches = allStations.filter(s => 
        s.name.toLowerCase().includes(query)
    ).slice(0, 8);
    
    if (matches.length === 0) {
        container.classList.remove('active');
        return;
    }
    
    container.innerHTML = matches.map(station => `
        <div class="trip-autocomplete-item" onclick="selectStation('${type}', '${station.name}', ${station.lat}, ${station.lon})">
            <div>
                <div class="station-name">${station.name}</div>
                <div class="station-lines">
                    ${station.lines.map(line => `
                        <span class="line-badge" style="background: ${getLineColor(line)}">${line.replace('Green-', 'GL-')}</span>
                    `).join('')}
                </div>
            </div>
        </div>
    `).join('');
    
    container.classList.add('active');
}

function hideAutocomplete(type) {
    const container = document.getElementById(`${type}-autocomplete`);
    if (container) container.classList.remove('active');
}

function selectStation(type, name, lat, lon) {
    const input = document.getElementById(type === 'origin' ? 'trip-origin' : 'trip-destination');
    if (input) input.value = name;
    
    if (type === 'origin') {
        selectedOrigin = { name, lat, lon };
    } else {
        selectedDestination = { name, lat, lon };
    }
    
    hideAutocomplete(type);
}

function useCurrentLocationForTrip() {
    if (userLocation && userLocation.lat && userLocation.lon) {
        const input = document.getElementById('trip-origin');
        if (input) {
            if (userLocation.station) {
                input.value = userLocation.station;
                selectedOrigin = { 
                    name: userLocation.station, 
                    lat: userLocation.lat, 
                    lon: userLocation.lon 
                };
            } else {
                input.value = 'Current Location';
                selectedOrigin = { 
                    name: 'Current Location', 
                    lat: userLocation.lat, 
                    lon: userLocation.lon,
                    coords: `${userLocation.lat},${userLocation.lon}`
                };
            }
        }
    } else {
        requestLocation();
        showNotification('Please enable location access', 'info');
    }
}

function swapTripInputs() {
    const originInput = document.getElementById('trip-origin');
    const destInput = document.getElementById('trip-destination');
    
    if (originInput && destInput) {
        const tempValue = originInput.value;
        originInput.value = destInput.value;
        destInput.value = tempValue;
        
        const tempSelected = selectedOrigin;
        selectedOrigin = selectedDestination;
        selectedDestination = tempSelected;
    }
}

async function searchTrip() {
    const originInput = document.getElementById('trip-origin');
    const destInput = document.getElementById('trip-destination');
    const resultsContainer = document.getElementById('trip-results');
    
    const origin = originInput?.value?.trim();
    const destination = destInput?.value?.trim();
    
    if (!origin || !destination) {
        showNotification('Please enter both origin and destination', 'error');
        return;
    }
    
    // Show loading
    resultsContainer.innerHTML = `
        <div class="trip-loading">
            <div class="spinner"></div>
            <div>Finding best routes...</div>
        </div>
    `;
    
    try {
        const requestBody = {
            origin: origin,
            destination: destination
        };
        
        // Add coordinates if available
        if (selectedOrigin?.coords) {
            requestBody.origin_coords = selectedOrigin.coords;
        } else if (selectedOrigin?.lat) {
            requestBody.origin_coords = `${selectedOrigin.lat},${selectedOrigin.lon}`;
        }
        
        if (selectedDestination?.lat) {
            requestBody.dest_coords = `${selectedDestination.lat},${selectedDestination.lon}`;
        }
        
        const response = await fetch('/api/trip-planner', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultsContainer.innerHTML = `<div class="trip-error">${data.error}</div>`;
            return;
        }
        
        if (!data.routes || data.routes.length === 0) {
            resultsContainer.innerHTML = `
                <div class="trip-error">
                    No transit routes found. Try different locations or check if transit is available.
                </div>
            `;
            return;
        }
        
        displayTripResults(data.routes);
        
    } catch (error) {
        console.error('Trip planner error:', error);
        resultsContainer.innerHTML = `<div class="trip-error">Failed to get directions. Please try again.</div>`;
    }
}

function displayTripResults(routes) {
    const container = document.getElementById('trip-results');
    
    container.innerHTML = `
        <div class="trip-routes">
            ${routes.map((route, index) => `
                <div class="trip-route-card" onclick="toggleRouteDetails(this)">
                    <div class="trip-route-header">
                        <div class="trip-route-time">
                            <span class="trip-route-duration">${route.duration_text || route.duration + ' min'}</span>
                            <span class="trip-route-range">${route.time_range || ''}</span>
                        </div>
                        <div class="trip-route-steps">
                            ${route.steps.map((step, i) => `
                                ${i > 0 ? '<div class="trip-step-connector"></div>' : ''}
                                <div class="trip-step-badge ${step.type}" style="${step.color ? `--badge-color: ${step.color}; background: ${step.color}` : ''}">
                                    ${getStepIcon(step.type)}
                                    ${step.short_name || step.name}
                                </div>
                            `).join('')}
                        </div>
                        <div class="trip-route-info">
                            <span class="trip-route-transfers">${route.transfers === 0 ? 'Direct' : route.transfers + ' transfer' + (route.transfers > 1 ? 's' : '')}</span>
                            ${route.fare ? `<span class="trip-route-fare">${route.fare}</span>` : ''}
                            ${route.walk_time ? `
                                <div class="trip-walk-time">
                                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M13.5 5.5c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zM9.8 8.9L7 23h2.1l1.8-8 2.1 2v6h2v-7.5l-2.1-2 .6-3C14.8 12 16.8 13 19 13v-2c-1.9 0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1L6 8.3V13h2V9.6l1.8-.7"/></svg>
                                    ${route.walk_time} min walk
                                </div>
                            ` : ''}
                        </div>
                        <div class="trip-route-expand">
                            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M6 9l6 6 6-6"/>
                            </svg>
                        </div>
                    </div>
                    <div class="trip-route-details">
                        ${route.detailed_steps.map((step, i) => `
                            <div class="trip-detail-step" style="--step-color: ${step.color || 'var(--border)'}">
                                <div class="trip-detail-icon ${step.type}">
                                    ${step.type === 'walk' ? '🚶' : step.type === 'transit' ? '🚇' : '📍'}
                                </div>
                                <div class="trip-detail-content">
                                    <div class="trip-detail-instruction">${step.instruction}</div>
                                    <div class="trip-detail-info">${step.details || ''}</div>
                                    ${step.time ? `<div class="trip-detail-time">${step.time}${step.duration ? ' • ' + step.duration : ''}</div>` : ''}
                                    ${step.num_stops ? `<div class="trip-detail-info">${step.num_stops} stop${step.num_stops !== 1 ? 's' : ''}</div>` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function getStepIcon(type) {
    const icons = {
        'subway': '🚇',
        'bus': '🚌',
        'commuter-rail': '🚂',
        'silver-line': '🚌',
        'ferry': '⛴️',
        'tram': '🚊',
        'walk': '🚶',
        'transit': '🚇'
    };
    return icons[type] || '🚇';
}

function toggleRouteDetails(card) {
    card.classList.toggle('expanded');
}

function getLineColor(line) {
    const colors = {
        'Red': '#DA291C',
        'Orange': '#ED8B00',
        'Blue': '#003DA5',
        'Green-B': '#00843D',
        'Green-C': '#00843D',
        'Green-D': '#00843D',
        'Green-E': '#00843D',
        'Green': '#00843D',
        'Mattapan': '#DA291C',
        'Silver': '#7C878E'
    };
    return colors[line] || '#666666';
}

// Initialize trip planner when page loads
document.addEventListener('DOMContentLoaded', () => {
    setupTripPlanner();
});

// Also setup when switching to overview tab
const originalSwitchTab = window.switchTab;
if (typeof originalSwitchTab === 'function') {
    window.switchTab = function(tabName) {
        originalSwitchTab(tabName);
        if (tabName === 'overview') {
            setupTripPlanner();
        }
    };
}
