document.addEventListener('DOMContentLoaded', () => {
    setupUploadZone();
    setupNavigation();
});

let charts = {};
let allAlertsData = [];
let rawLogsData = [];

// === UI & NAVIGATION ===
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Toggle active nav styling
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active section
            const targetId = btn.getAttribute('data-target');
            document.querySelectorAll('.view-panel').forEach(panel => {
                panel.classList.remove('active-view');
            });
            document.getElementById(targetId).classList.add('active-view');
            
            // Close drawer if switching tabs
            closeDrawer();
        });
    });
}

function removeEmptyStates() {
    document.querySelectorAll('.empty-state-container').forEach(container => {
        container.classList.remove('is-empty');
    });
}

function setEmptyStates() {
    document.querySelectorAll('.empty-state-container').forEach(container => {
        container.classList.add('is-empty');
    });
}

// Set initial empty state on load
setEmptyStates();

// === UPLOAD HANDLING ===
function setupUploadZone() {
    const uploadCard = document.querySelector('.upload-card');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadCard.addEventListener(eventName, () => uploadCard.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, () => uploadCard.classList.remove('dragover'), false);
    });

    uploadCard.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function() { handleFiles(this.files); });
}

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    if (!file.name.endsWith('.csv')) { alert("Please upload a .csv file."); return; }
    
    document.getElementById('uploadSection').classList.remove('active-view');
    document.getElementById('loadingSection').classList.add('active-view');
    
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') { loadInvestigationData(); } 
        else { alert('Error: ' + data.error); resetUpload(); }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during upload.');
        resetUpload();
    });
}

function resetUpload() {
    document.getElementById('loadingSection').classList.remove('active-view');
    document.getElementById('uploadSection').classList.add('active-view');
}

// === DATA LOADING ===
function loadInvestigationData() {
    const ts = new Date().getTime();
    
    // Load Events Data
    Papa.parse(`/data/alerts/all_alerts.csv?t=${ts}`, {
        download: true, header: true, skipEmptyLines: true,
        complete: function(res1) {
            allAlertsData = res1.data.reverse();
            
            // Load Raw Telemetry Data
            Papa.parse(`/data/processed/clean_logs.csv?t=${ts}`, {
                download: true, header: true, skipEmptyLines: true,
                complete: function(res2) {
                    rawLogsData = res2.data.reverse();
                    
                    removeEmptyStates();
                    processEvents();
                    processRawLogs();
                    
                    // Automatically switch to Events tab after load
                    document.querySelector('.nav-btn[data-target="eventsSection"]').click();
                    document.getElementById('loadingSection').classList.remove('active-view');
                },
                error: function() { console.error("Error loading clean_logs"); }
            });
        }
    });
}

// === EVENTS PROCESSING ===
function processEvents() {
    if (!allAlertsData || allAlertsData.length === 0) {
        setEmptyStates();
        return;
    }

    let ruleCount = 0;
    let mlCount = 0;
    let hybridCount = 0;
    let actionableCount = 0;
    
    const actionableEntities = {}; // Group for Threat Response

    const timeCounts = {};

    allAlertsData.forEach(alert => {
        // Fix Stats - count total fires
        const src = (alert.source || '');
        const type = (alert.alert_type || '');
        
        if (src === 'Rule-based') ruleCount++;
        if (type === 'Unified ML anomaly') {
            try {
                const match = (alert.details || '').match(/'action_count':\s*([0-9.]+)/);
                if (match && match[1]) {
                    mlCount += parseInt(match[1], 10);
                } else {
                    mlCount++;
                }
            } catch(e) {
                mlCount++;
            }
        }
        if (src === 'Hybrid') hybridCount++;
        
        // Actionable / High Severity
        const sev = getSeverity(alert);
        if (sev.label === 'HIGH') {
            actionableCount++;
            
            // Group by user (or host if user missing) for the Threat Response panel
            const entityKey = alert.user && alert.user !== '-' ? alert.user : (alert.host || 'Unknown');
            if (!actionableEntities[entityKey]) actionableEntities[entityKey] = [];
            actionableEntities[entityKey].push(alert);
        }
        
        // Timeline Prep
        const t = alert.timestamp || alert.time_window;
        if (t) {
            const date = new Date(t);
            if (!isNaN(date)) {
                const hourStr = `${date.getMonth()+1}/${date.getDate()} ${String(date.getHours()).padStart(2, '0')}:00`;
                timeCounts[hourStr] = (timeCounts[hourStr] || 0) + 1;
            }
        }
    });

    document.getElementById('ruleCount').textContent = ruleCount;
    document.getElementById('mlCount').textContent = mlCount;
    document.getElementById('hybridCount').textContent = hybridCount;
    document.getElementById('actionableCount').textContent = actionableCount;

    // Populate Threat Response Panel
    const responseContainer = document.getElementById('threat-response-container');
    if (responseContainer) {
        responseContainer.innerHTML = '';
        if (Object.keys(actionableEntities).length === 0) {
            responseContainer.innerHTML = '<div class="empty-state-message">No actionable threats at this time.</div>';
        } else {
            Object.keys(actionableEntities).forEach(entity => {
                const alertsList = actionableEntities[entity];
                const card = document.createElement('div');
                card.className = 'drawer-card';
                card.style.background = 'rgba(255, 51, 102, 0.05)';
                card.style.border = '1px solid rgba(255, 51, 102, 0.2)';
                card.style.marginBottom = '1rem';
                
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3 style="color: #fff; margin: 0; font-size: 1.1rem;">Compromised Entity: <span style="color: #ff3366;">${entity}</span></h3>
                        <span class="badge" style="background: rgba(255, 51, 102, 0.2); color: #ff3366;">${alertsList.length} High Severity Alerts</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 1rem;">This entity has triggered multiple high-severity lateral movement forensic rules. Immediate SOC intervention is recommended.</p>
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="nav-btn active" style="font-size: 0.8rem; padding: 0.5rem 1rem; width: auto;" onclick="alert('Host Isolated from Network: ${entity}')">🛑 Isolate Host</button>
                        <button class="nav-btn active" style="font-size: 0.8rem; padding: 0.5rem 1rem; width: auto;" onclick="alert('Active Directory Account Disabled: ${entity}')">🔒 Disable Account</button>
                        <button class="nav-btn" style="font-size: 0.8rem; padding: 0.5rem 1rem; width: auto;" onclick="alert('EDR Scan Triggered for: ${entity}')">🔍 Trigger EDR Scan</button>
                    </div>
                `;
                responseContainer.appendChild(card);
            });
        }
    }

    renderLineChart('alertsTimelineChart', timeCounts, 'Event Volume');

    // Populate Table
    const tbody = document.querySelector('#alertsTable tbody');
    tbody.innerHTML = '';

    allAlertsData.slice(0, 100).forEach((alert) => {
        const tr = document.createElement('tr');
        const sev = getSeverity(alert);
        const date = new Date(alert.timestamp || alert.time_window);
        const formattedDate = !isNaN(date.getTime()) ? date.toLocaleString() : (alert.timestamp || alert.time_window);
        
        // Infer Event ID for display based on alert type
        let evtId = '-';
        const typeStr = (alert.alert_type || '').toLowerCase();
        if (typeStr.includes('process')) evtId = '1';
        else if (typeStr.includes('network') || typeStr.includes('hosts')) evtId = '3';
        else if (typeStr.includes('login')) evtId = '4624';
        
        tr.innerHTML = `
            <td>${formattedDate}</td>
            <td>${evtId}</td>
            <td>${alert.user || '-'}</td>
            <td>${alert.host || '-'}</td>
            <td>${alert.process_name || '-'}</td>
            <td>${alert.source_ip || '-'}</td>
            <td>${alert.source_port || '-'}</td>
            <td>${alert.dest_ip || '-'}</td>
            <td>${alert.dest_port || '-'}</td>
            <td><span class="detection-source-badge ${alert.source === 'ML' ? 'src-ml' : (alert.source === 'Hybrid' ? 'src-hybrid' : 'src-rule')}">${alert.source || '-'}</span></td>
            <td>${alert.alert_type === 'Unified ML anomaly' && alert.source === 'Hybrid' ? 'ML Behavior + Rule Confirmation' : (alert.alert_type || '-')}</td>
            <td><span class="severity-indicator ${sev.cls}">${sev.label}</span></td>
        `;
        tr.addEventListener('click', () => openAlertDrawer(alert));
        tbody.appendChild(tr);
    });
}

// === RAW LOGS PROCESSING ===
function processRawLogs() {
    if (!rawLogsData || rawLogsData.length === 0) return;

    const tbody = document.querySelector('#rawLogsTable tbody');
    tbody.innerHTML = '';
    
    // Only display first 200 for performance
    rawLogsData.slice(0, 200).forEach((log) => {
        const tr = document.createElement('tr');
        
        const date = new Date(log.timestamp);
        const formattedDate = !isNaN(date.getTime()) ? date.toLocaleString() : log.timestamp;
        
        const cmdOrAction = log.command_line !== '-' ? log.command_line : (log.dest_port !== '-' ? `Network Connection -> Port ${log.dest_port}` : 'Background process');

        tr.innerHTML = `
            <td>${formattedDate}</td>
            <td>${log.event_id || '-'}</td>
            <td>${log.user || '-'}</td>
            <td>${log.host || '-'}</td>
            <td>${cmdOrAction.substring(0, 80)}${cmdOrAction.length > 80 ? '...' : ''}</td>
        `;
        tr.addEventListener('click', () => openRawLogDrawer(log));
        tbody.appendChild(tr);
    });
}

// === DRAWER & INVESTIGATION UTILS ===
function getSeverity(alert) {
    if (alert.source === 'ML' || alert.source === 'Hybrid' || (alert.alert_type && alert.alert_type.toLowerCase().includes('multiple hosts'))) {
        return { label: 'HIGH', cls: 'sev-high' };
    }
    return { label: 'MEDIUM', cls: 'sev-med' };
}

function openAlertDrawer(alert) {
    const drawer = document.getElementById('investigationDrawer');
    
    // Populate Headers
    const sev = getSeverity(alert);
    const sevBadge = document.getElementById('drawer-severity');
    sevBadge.textContent = sev.label;
    sevBadge.className = `severity-indicator ${sev.cls}`;
    
    document.getElementById('drawer-alert-type').textContent = alert.alert_type || 'Unknown Detection';
    
    const srcBadge = document.getElementById('drawer-source-badge');
    const source = alert.source || 'Unknown';
    srcBadge.textContent = source.toUpperCase() + ' DETECTION';
    if (source === 'ML') srcBadge.className = 'detection-source-badge src-ml';
    else if (source === 'Hybrid') srcBadge.className = 'detection-source-badge src-hybrid';
    else srcBadge.className = 'detection-source-badge src-rule';

    // Populate Metadata
    const date = new Date(alert.timestamp || alert.time_window);
    document.getElementById('drawer-time').textContent = !isNaN(date.getTime()) ? date.toLocaleString() : (alert.timestamp || alert.time_window);
    document.getElementById('drawer-user').textContent = alert.user || '-';
    document.getElementById('drawer-host').textContent = alert.host || '-';
    document.getElementById('drawer-src').textContent = alert.source_ip !== '-' ? alert.source_ip : '-';
    document.getElementById('drawer-dst').textContent = alert.dest_ip !== '-' ? alert.dest_ip : '-';
    document.getElementById('drawer-pid').textContent = '-';

    // Generate Explanations
    generateForensicExplanations(alert, sev.label);

    drawer.classList.add('open');
}

function openRawLogDrawer(log) {
    const drawer = document.getElementById('investigationDrawer');
    
    // Headers for raw log
    document.getElementById('drawer-severity').textContent = 'INFO';
    document.getElementById('drawer-severity').className = 'severity-indicator sev-low';
    document.getElementById('drawer-alert-type').textContent = `Sysmon Event ID: ${log.event_id}`;
    
    const srcBadge = document.getElementById('drawer-source-badge');
    srcBadge.textContent = 'RAW TELEMETRY';
    srcBadge.className = 'detection-source-badge src-rule';

    // Metadata
    const date = new Date(log.timestamp);
    document.getElementById('drawer-time').textContent = !isNaN(date.getTime()) ? date.toLocaleString() : log.timestamp;
    document.getElementById('drawer-user').textContent = log.user || '-';
    document.getElementById('drawer-host').textContent = log.host || '-';
    document.getElementById('drawer-src').textContent = log.source_ip !== '-' ? log.source_ip : '-';
    document.getElementById('drawer-dst').textContent = log.dest_ip !== '-' ? log.dest_ip : '-';
    document.getElementById('drawer-pid').textContent = log.process_id || '-';

    // Clear explanations
    document.getElementById('drawer-mitre').innerHTML = '<span class="text-muted">N/A for raw telemetry</span>';
    document.getElementById('drawer-severity-container').style.display = 'none';
    document.getElementById('drawer-detection-label').textContent = 'Process Context';
    document.getElementById('drawer-explanation').textContent = `Raw Sysmon log ingestion. Process: ${log.process_name || '-'}\nCommand Line: ${log.command_line || '-'}`;

    drawer.classList.add('open');
}

function closeDrawer() {
    document.getElementById('investigationDrawer').classList.remove('open');
}

function generateForensicExplanations(alert, severityLabel) {
    document.getElementById('drawer-severity-container').style.display = 'block';
    const typeStr = (alert.alert_type || '').toLowerCase();
    
    let sevExpl = "";
    if (severityLabel === 'HIGH') {
        sevExpl = "Classified as HIGH severity due to indicators highly correlated with active lateral movement or credential dumping, posing an immediate risk.";
    } else {
        sevExpl = "Classified as MEDIUM severity because while the behavior is anomalous, it may be attributed to administrative tasks.";
    }
    document.getElementById('drawer-severity-expl').textContent = sevExpl;

    let explanation = "";
    let mitreTags = [];

    if (alert.source === 'Hybrid') {
        document.getElementById('drawer-detection-label').textContent = 'True Hybrid Confirmation';
        mitreTags = ['T1021 - Remote Services', 'T1059 - Command and Scripting'];
        explanation = `ULTRA HIGH CONFIDENCE: This event was flagged by BOTH the static rule engine and the ML anomaly detection engine within the same time window. \n\nThe rules confirmed explicit suspicious activity, and the ML verified that this behavior represents a massive volumetric and structural deviation from the user's baseline.`;
    } else if (alert.source === 'ML') {
        document.getElementById('drawer-detection-label').textContent = 'ML Detection Reasoning';
        mitreTags = ['T1021 - Remote Services', 'T1059 - Command and Scripting'];
        explanation = `The unsupervised ML model flagged this window as highly anomalous. \n\nBehavioral Reasoning: The event volume, hosts targeted, or execution characteristics deviate significantly from established baselines. Specifically, the model detected structural anomalies fused with a volumetric spike.`;
    } else {
        document.getElementById('drawer-detection-label').textContent = 'Rule Detection Reasoning';
        if (typeStr.includes('remote process')) {
            mitreTags = ['T1059 - Command and Scripting', 'T1047 - WMI'];
            explanation = `Static forensic rule matched: Suspicious process creation. Command line arguments indicate remote execution via WMI (wmic.exe) or PsExec.`;
        } else if (typeStr.includes('multiple hosts')) {
            mitreTags = ['T1021 - Remote Services', 'T1078 - Valid Accounts'];
            explanation = `Static forensic rule matched: Rapid lateral movement. The user authenticated to multiple distinct hosts within a very narrow time window.`;
        } else if (typeStr.includes('sensitive host')) {
            mitreTags = ['T1078 - Valid Accounts'];
            explanation = `Static forensic rule matched: Authentication targeted a highly sensitive host by a user who does not typically exhibit this access pattern.`;
        } else if (typeStr.includes('working hours')) {
            mitreTags = ['T1078 - Valid Accounts'];
            explanation = `Static forensic rule matched: The authentication occurred significantly outside standard operational hours.`;
        } else {
            explanation = "Static forensic rule matched based on predefined IOCs.";
        }
    }

    const mitreContainer = document.getElementById('drawer-mitre');
    mitreContainer.innerHTML = '';
    if (mitreTags.length > 0) {
        mitreTags.forEach(tag => {
            const span = document.createElement('span');
            span.className = 'mitre-tag';
            span.textContent = tag;
            mitreContainer.appendChild(span);
        });
    }
    
    document.getElementById('drawer-explanation').textContent = explanation;
}

// === CHARTS ===
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

function renderLineChart(canvasId, dataObj, label) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (charts[canvasId]) charts[canvasId].destroy();

    const sortedLabels = Object.keys(dataObj).sort();
    
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: label,
                data: sortedLabels.map(k => dataObj[k]),
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.15)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#06b6d4',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#06b6d4'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(15, 16, 19, 0.9)', titleColor: '#06b6d4' } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { stepSize: 1 } },
                x: { grid: { color: 'rgba(255, 255, 255, 0.02)' } }
            }
        }
    });
}
