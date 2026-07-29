/**
 * WEBLEA - Frontend Logic
 * Layered Elastic Analysis Web Application
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    unitSystem: 'imperial',
    mode: 'heatmap',
    profileType: 'depth',
    layers: [],
    loads: [],
    heatmapData: null,
    results: null,
    // Cached data from last analysis (all 13 responses)
    cachedHeatmapData: null,
    cachedHeatmapX: null,
    cachedHeatmapZ: null,
    cachedLayerBoundaries: null,
    currentResponse: 'eps_z',
    // Separate cache for pointwise results
    cachedPointwiseData: null
};

// Unit configurations
const UNITS = {
    imperial: {
        length: 'in',
        stress: 'psi',
        force: 'lbs',
        modulus: 'psi'
    },
    si: {
        length: 'mm',
        stress: 'MPa',
        force: 'kN',
        modulus: 'MPa'
    }
};

// Conversion factors (imperial to SI)
const CONVERSIONS = {
    length: 25.4,        // in to mm
    force: 0.00444822,   // lbs to kN
    modulus: 0.00689476  // psi to MPa
};

// Default values
const DEFAULTS = {
    imperial: {
        layers: [
            { thickness: 6, modulus: 500000, poisson: 0.35 },
            { thickness: 18, modulus: 50000, poisson: 0.40 },
            { thickness: null, modulus: 10000, poisson: 0.45 }
        ],
        loads: [
            { magnitude: 9000, x: 10 },
            { magnitude: 9000, x: 20 }
        ],
        contactRadius: 4
    },
    si: {
        layers: [
            { thickness: 152, modulus: 3447, poisson: 0.35 },
            { thickness: 457, modulus: 345, poisson: 0.40 },
            { thickness: null, modulus: 69, poisson: 0.45 }
        ],
        loads: [
            { magnitude: 40, x: 254 },
            { magnitude: 40, x: 508 }
        ],
        contactRadius: 102
    }
};

// Response names for display
const RESPONSE_NAMES = {
    'deflection_z': 'Vertical Deflection',
    'eps_x': 'εx',
    'eps_y': 'εy',
    'eps_z': 'εz',
    'eps_xy': 'γxy',
    'eps_yz': 'γyz',
    'eps_xz': 'γxz',
    'sigma_x': 'σx',
    'sigma_y': 'σy',
    'sigma_z': 'σz',
    'sigma_xy': 'τxy',
    'sigma_yz': 'τyz',
    'sigma_xz': 'τxz'
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    loadDefaults();
    setupEventListeners();
    renderLayers();
    renderLoads();
    updateUnitLabels();
    updateGridSummary();
}

function loadDefaults() {
    const defaults = DEFAULTS[state.unitSystem];
    state.layers = JSON.parse(JSON.stringify(defaults.layers));
    state.loads = JSON.parse(JSON.stringify(defaults.loads));
    document.getElementById('contactRadius').value = defaults.contactRadius;
}

function setupEventListeners() {
    // Unit selector
    document.getElementById('unitSystem').addEventListener('change', handleUnitChange);

    // Layer/Load buttons
    document.getElementById('addLayerBtn').addEventListener('click', addLayer);
    document.getElementById('addLoadBtn').addEventListener('click', addLoad);

    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => handleTabChange(btn.dataset.tab));
    });

    // Note: profileType removed from lineplot tab - now using lineplotProfileType in results header

    // Pointwise subtabs
    document.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.addEventListener('click', () => handleSubtabChange(btn.dataset.subtab));
    });

    // Add point button
    document.getElementById('addPointBtn').addEventListener('click', addPoint);

    // Grid generator
    document.getElementById('generateGridBtn').addEventListener('click', generateGridPoints);
    document.getElementById('gridXValues').addEventListener('input', updateGridSummary);
    document.getElementById('gridZValues').addEventListener('input', updateGridSummary);

    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);

    // Results actions
    document.getElementById('copyResultsBtn').addEventListener('click', copyResults);
    document.getElementById('downloadCsvBtn').addEventListener('click', downloadCSV);

    // Response selector for cached heatmap display
    document.getElementById('displayResponseType').addEventListener('change', (e) => {
        state.currentResponse = e.target.value;
        if (state.cachedHeatmapData && state.mode === 'heatmap') {
            renderCachedHeatmap();
        }
    });

    // Line plot position slider - instant update
    const positionSlider = document.getElementById('positionSlider');
    if (positionSlider) {
        positionSlider.addEventListener('input', (e) => {
            updateSliderLabel();
            if (state.cachedHeatmapData && state.mode === 'lineplot') {
                renderLinePlotFromCache();
            }
        });
    }

    // Line plot profile type toggle
    const lineplotProfileType = document.getElementById('lineplotProfileType');
    if (lineplotProfileType) {
        lineplotProfileType.addEventListener('change', () => {
            populateSliderFromCache();
            if (state.cachedHeatmapData && state.mode === 'lineplot') {
                renderLinePlotFromCache();
            }
        });
    }

    // Line plot response checkboxes - auto-update plot when changed
    document.querySelectorAll('#lineplotResponses input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', () => {
            if (state.cachedHeatmapData && state.mode === 'lineplot') {
                renderLinePlotFromCache();
            }
        });
    });

    // Heatmap interpolation checkbox - auto-update when toggled
    const interpolateCheckbox = document.getElementById('heatmapInterpolate');
    if (interpolateCheckbox) {
        interpolateCheckbox.addEventListener('change', () => {
            if (state.cachedHeatmapData && state.mode === 'heatmap') {
                renderCachedHeatmap();
            }
        });
    }

    // Toast close
    document.querySelector('.toast-close').addEventListener('click', hideToast);
}

// ============================================================================
// Layer Management
// ============================================================================

function renderLayers() {
    const tbody = document.getElementById('layersTableBody');
    tbody.innerHTML = '';

    state.layers.forEach((layer, index) => {
        const isSubgrade = index === state.layers.length - 1;
        const tr = document.createElement('tr');

        const layerName = isSubgrade ? 'Subgrade' : `Layer ${index + 1}`;
        const nameClass = isSubgrade ? 'layer-name subgrade' : 'layer-name';

        tr.innerHTML = `
            <td><span class="${nameClass}">${layerName}</span></td>
            <td>
                <input type="number" 
                       value="${layer.thickness ?? ''}" 
                       ${isSubgrade ? 'disabled placeholder="∞"' : 'min="0.1" step="0.1"'}
                       data-field="thickness" data-index="${index}">
            </td>
            <td>
                <input type="number" 
                       value="${layer.modulus}" 
                       min="1" step="100"
                       data-field="modulus" data-index="${index}">
            </td>
            <td>
                <input type="number" 
                       value="${layer.poisson}" 
                       min="0" max="0.5" step="0.01"
                       data-field="poisson" data-index="${index}">
            </td>
            <td>
                <button class="btn-remove" 
                        ${state.layers.length <= 2 ? 'disabled' : ''} 
                        data-index="${index}"
                        title="Remove">×</button>
            </td>
        `;

        // Event listeners
        tr.querySelectorAll('input').forEach(input => {
            input.addEventListener('change', handleLayerChange);
        });

        tr.querySelector('.btn-remove').addEventListener('click', () => removeLayer(index));

        tbody.appendChild(tr);
    });
}

function handleLayerChange(e) {
    const index = parseInt(e.target.dataset.index);
    const field = e.target.dataset.field;
    let value = e.target.value;

    if (field === 'thickness') {
        state.layers[index][field] = value ? parseFloat(value) : null;
    } else if (field === 'poisson') {
        state.layers[index][field] = Math.min(0.5, Math.max(0, parseFloat(value) || 0));
        e.target.value = state.layers[index][field];
    } else {
        state.layers[index][field] = parseFloat(value) || 0;
    }
}

function addLayer() {
    const newLayer = { thickness: 6, modulus: 50000, poisson: 0.35 };
    if (state.unitSystem === 'si') {
        newLayer.thickness = 150;
        newLayer.modulus = 345;
    }
    state.layers.splice(state.layers.length - 1, 0, newLayer);
    renderLayers();
}

function removeLayer(index) {
    if (state.layers.length <= 2) return;
    state.layers.splice(index, 1);
    renderLayers();
}

// ============================================================================
// Load Management
// ============================================================================

function renderLoads() {
    const tbody = document.getElementById('loadsTableBody');
    tbody.innerHTML = '';

    state.loads.forEach((load, index) => {
        const tr = document.createElement('tr');

        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>
                <input type="number" 
                       value="${load.magnitude}" 
                       min="0" step="100"
                       data-field="magnitude" data-index="${index}">
            </td>
            <td>
                <input type="number" 
                       value="${load.x}" 
                       step="1"
                       data-field="x" data-index="${index}">
            </td>
            <td>
                <button class="btn-remove" 
                        ${state.loads.length <= 1 ? 'disabled' : ''} 
                        data-index="${index}"
                        title="Remove">×</button>
            </td>
        `;

        tr.querySelectorAll('input').forEach(input => {
            input.addEventListener('change', handleLoadChange);
        });

        tr.querySelector('.btn-remove').addEventListener('click', () => removeLoad(index));

        tbody.appendChild(tr);
    });
}

function handleLoadChange(e) {
    const index = parseInt(e.target.dataset.index);
    const field = e.target.dataset.field;
    state.loads[index][field] = parseFloat(e.target.value) || 0;
}

function addLoad() {
    const lastLoad = state.loads[state.loads.length - 1];
    const spacing = state.unitSystem === 'imperial' ? 10 : 254;
    state.loads.push({
        magnitude: lastLoad.magnitude,
        x: lastLoad.x + spacing
    });
    renderLoads();
}

function removeLoad(index) {
    if (state.loads.length <= 1) return;
    state.loads.splice(index, 1);
    renderLoads();
}

// ============================================================================
// Unit Handling
// ============================================================================

function handleUnitChange(e) {
    const newSystem = e.target.value;
    if (newSystem === state.unitSystem) return;

    // Convert current values
    convertUnits(state.unitSystem, newSystem);
    state.unitSystem = newSystem;

    renderLayers();
    renderLoads();
    updateUnitLabels();

    // Also update range inputs
    convertRangeInputs(state.unitSystem === 'si');
}

function convertUnits(from, to) {
    const toSI = (to === 'si');
    const factor = toSI ? 1 : -1;

    // Convert layers
    state.layers.forEach(layer => {
        if (layer.thickness !== null) {
            layer.thickness = convertValue(layer.thickness, 'length', toSI);
        }
        layer.modulus = convertValue(layer.modulus, 'modulus', toSI);
    });

    // Convert loads
    state.loads.forEach(load => {
        load.magnitude = convertValue(load.magnitude, 'force', toSI);
        load.x = convertValue(load.x, 'length', toSI);
    });

    // Convert contact radius
    const radiusInput = document.getElementById('contactRadius');
    radiusInput.value = convertValue(parseFloat(radiusInput.value), 'length', toSI).toFixed(1);
}

function convertValue(value, type, toSI) {
    const factor = CONVERSIONS[type];
    if (toSI) {
        return Math.round(value * factor * 100) / 100;
    } else {
        return Math.round(value / factor * 100) / 100;
    }
}

function convertRangeInputs(toSI) {
    const rangeIds = ['xMin', 'xMax', 'zMin', 'zMax', 'profileRangeMin', 'profileRangeMax', 'fixedValue'];
    rangeIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.value = convertValue(parseFloat(el.value) || 0, 'length', toSI).toFixed(0);
        }
    });

    // Also convert resolution - keep approximately the same physical size
    const resEl = document.getElementById('resolution');
    if (resEl) {
        const oldVal = parseFloat(resEl.value) || 1;
        const newVal = convertValue(oldVal, 'length', toSI);
        // Round to sensible values
        resEl.value = toSI ? Math.round(newVal / 5) * 5 || 25 : Math.round(newVal * 2) / 2 || 1;
    }
}

function updateUnitLabels() {
    const units = UNITS[state.unitSystem];
    document.querySelectorAll('[data-unit]').forEach(el => {
        const unitType = el.dataset.unit;
        el.textContent = units[unitType] || '';
    });
}

// ============================================================================
// Tab Navigation
// ============================================================================

function handleTabChange(tab) {
    state.mode = tab;

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === tab + 'Tab');
    });

    // Show/hide response selector and lineplot controls based on mode
    const responseSelectorContainer = document.getElementById('responseSelectorContainer');
    const lineplotControls = document.getElementById('lineplotControls');

    if (!responseSelectorContainer) return;

    // Check if we have valid cached data (all required arrays must exist)
    const hasCachedData = state.cachedHeatmapData &&
        state.cachedHeatmapX &&
        state.cachedHeatmapZ &&
        state.cachedHeatmapX.length > 0 &&
        state.cachedHeatmapZ.length > 0;

    if (tab === 'heatmap' && hasCachedData) {
        responseSelectorContainer.classList.remove('hidden');
        if (lineplotControls) lineplotControls.classList.add('hidden');
        hideResultsTable();  // Hide pointwise table
        renderCachedHeatmap();
    } else if (tab === 'lineplot' && hasCachedData) {
        responseSelectorContainer.classList.add('hidden');
        if (lineplotControls) {
            lineplotControls.classList.remove('hidden');
            populateSliderFromCache();
        }
        hideResultsTable();  // Hide pointwise table
        renderLinePlotFromCache();
    } else if (tab === 'pointwise') {
        // Pointwise mode - hide heatmap/lineplot controls, restore cached pointwise results if any
        responseSelectorContainer.classList.add('hidden');
        if (lineplotControls) lineplotControls.classList.add('hidden');
        // Restore cached pointwise results if available
        if (state.cachedPointwiseData) {
            renderPointwiseFromCache();
        }
    } else {
        responseSelectorContainer.classList.add('hidden');
        if (lineplotControls) lineplotControls.classList.add('hidden');
    }
}

// Populate slider with cached grid values
function populateSliderFromCache() {
    const slider = document.getElementById('positionSlider');
    const profileTypeEl = document.getElementById('lineplotProfileType');
    if (!slider || !state.cachedHeatmapX || !state.cachedHeatmapZ) return;

    const profileType = profileTypeEl ? profileTypeEl.value : 'depth';
    const values = profileType === 'depth' ? state.cachedHeatmapX : state.cachedHeatmapZ;

    if (values && values.length > 0) {
        slider.min = 0;
        slider.max = values.length - 1;
        slider.value = 0;
        updateSliderLabel();
    }
}

// Update slider label with current value
function updateSliderLabel() {
    const slider = document.getElementById('positionSlider');
    const label = document.getElementById('sliderLabel');
    const profileTypeEl = document.getElementById('lineplotProfileType');
    if (!slider || !label) return;

    const profileType = profileTypeEl ? profileTypeEl.value : 'depth';
    const values = profileType === 'depth' ? state.cachedHeatmapX : state.cachedHeatmapZ;
    const idx = parseInt(slider.value) || 0;

    if (values && values[idx] !== undefined) {
        const axisLabel = profileType === 'depth' ? 'X' : 'Z';
        label.textContent = `${axisLabel} = ${values[idx].toFixed(1)}`;
    }
}

function handleProfileTypeChange() {
    const profileTypeEl = document.getElementById('profileType');
    if (!profileTypeEl) return;

    const type = profileTypeEl.value;
    state.profileType = type;

    const fixedLabel = document.getElementById('fixedValueLabel');
    const rangeLabel = document.getElementById('rangeLabel');

    if (!fixedLabel || !rangeLabel) return;

    if (type === 'depth') {
        fixedLabel.textContent = 'Fixed X:';
        rangeLabel.textContent = 'Z Range:';
    } else {
        fixedLabel.textContent = 'Fixed Z:';
        rangeLabel.textContent = 'X Range:';
    }
}

function handleSubtabChange(subtab) {
    document.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subtab === subtab);
    });

    document.getElementById('manualSubtab').classList.toggle('active', subtab === 'manual');
    document.getElementById('gridSubtab').classList.toggle('active', subtab === 'grid');
}

// ============================================================================
// Pointwise
// ============================================================================

function addPoint() {
    const container = document.getElementById('pointsContainer');
    const row = document.createElement('div');
    row.className = 'point-row';
    row.innerHTML = `
        <input type="number" class="point-x" placeholder="x" value="0">
        <input type="number" class="point-z" placeholder="z" value="0">
        <button class="btn-remove" title="Remove">×</button>
    `;
    row.querySelector('.btn-remove').addEventListener('click', () => row.remove());
    container.appendChild(row);
}

function getPoints() {
    const points = [];
    document.querySelectorAll('.point-row').forEach(row => {
        const x = parseFloat(row.querySelector('.point-x').value) || 0;
        const z = parseFloat(row.querySelector('.point-z').value) || 0;
        points.push({ x, z });
    });
    return points;
}

function parseCommaSeparated(str) {
    return str.split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n));
}

function updateGridSummary() {
    const xVals = parseCommaSeparated(document.getElementById('gridXValues').value);
    const zVals = parseCommaSeparated(document.getElementById('gridZValues').value);
    document.getElementById('gridSummary').textContent = `${xVals.length * zVals.length} points`;
}

function generateGridPoints() {
    const xVals = parseCommaSeparated(document.getElementById('gridXValues').value);
    const zVals = parseCommaSeparated(document.getElementById('gridZValues').value);

    const container = document.getElementById('pointsContainer');
    container.innerHTML = '';

    xVals.forEach(x => {
        zVals.forEach(z => {
            const row = document.createElement('div');
            row.className = 'point-row';
            row.innerHTML = `
                <input type="number" class="point-x" value="${x}">
                <input type="number" class="point-z" value="${z}">
                <button class="btn-remove" title="Remove">×</button>
            `;
            row.querySelector('.btn-remove').addEventListener('click', () => row.remove());
            container.appendChild(row);
        });
    });

    handleSubtabChange('manual');
    showToast(`Generated ${xVals.length * zVals.length} points`, 'success');
}

function getSelectedResponses(containerId) {
    const responses = [];
    document.querySelectorAll(`#${containerId} input:checked, .checkbox-group input:checked`).forEach(cb => {
        if (cb.closest('#' + containerId) || cb.closest('.tab-content.active')) {
            // Filter out checkboxes with no value or default 'on' (like Select All)
            if (cb.value && cb.value !== 'on') {
                responses.push(cb.value);
            }
        }
    });
    return responses.length > 0 ? responses : ['eps_z', 'sigma_z'];
}

// ============================================================================
// Analysis
// ============================================================================

async function runAnalysis() {
    const btn = document.getElementById('analyzeBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    btn.disabled = true;

    try {
        validateInputs();

        let endpoint, data;

        if (state.mode === 'heatmap') {
            endpoint = '/api/analyze';
            data = prepareHeatmapData();
        } else if (state.mode === 'lineplot') {
            endpoint = '/api/analyze-profile';
            data = prepareProfileData();
        } else {
            endpoint = '/api/analyze-points';
            data = preparePointwiseData();
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Analysis failed');
        }

        state.results = result;

        if (state.mode === 'heatmap') {
            state.heatmapData = result;
            renderHeatmap(result);
        } else if (state.mode === 'lineplot') {
            renderProfile(result);
        } else {
            renderPointwiseResults(result);
        }

    } catch (error) {
        showToast(error.message, 'error');
        console.error(error);
    } finally {
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
        btn.disabled = false;
    }
}

function validateInputs() {
    for (let i = 0; i < state.layers.length; i++) {
        const layer = state.layers[i];
        const isSubgrade = i === state.layers.length - 1;

        if (!isSubgrade && (!layer.thickness || layer.thickness <= 0)) {
            throw new Error(`Layer ${i + 1}: Thickness must be > 0`);
        }
        if (!layer.modulus || layer.modulus <= 0) {
            throw new Error(`${isSubgrade ? 'Subgrade' : 'Layer ' + (i + 1)}: Modulus must be > 0`);
        }
        if (layer.poisson < 0 || layer.poisson > 0.5) {
            throw new Error(`${isSubgrade ? 'Subgrade' : 'Layer ' + (i + 1)}: Poisson's ratio must be 0-0.5`);
        }
    }

    for (let i = 0; i < state.loads.length; i++) {
        if (!state.loads[i].magnitude || state.loads[i].magnitude <= 0) {
            throw new Error(`Load ${i + 1}: Magnitude must be > 0`);
        }
    }

    const radiusEl = document.getElementById('contactRadius');
    const radius = radiusEl ? parseFloat(radiusEl.value) : 0;
    if (!radius || radius <= 0) {
        throw new Error('Contact radius must be > 0');
    }
}

function prepareHeatmapData() {
    // Get elements with null guards and fallback defaults
    const getVal = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? parseFloat(el.value) || fallback : fallback;
    };

    return {
        layers: state.layers,
        loads: state.loads,
        contactRadius: getVal('contactRadius', 4),
        xMin: getVal('xMin', 0),
        xMax: getVal('xMax', 30),
        zMin: getVal('zMin', 0),
        zMax: getVal('zMax', 30),
        resolution: getVal('resolution', 1),
        unitSystem: state.unitSystem
        // Note: selectedResponse removed - backend now returns ALL responses
    };
}

function prepareProfileData() {
    const responses = [];
    document.querySelectorAll('#lineplotTab .checkbox input:checked').forEach(cb => {
        responses.push(cb.value);
    });

    return {
        layers: state.layers,
        loads: state.loads,
        contactRadius: getValSafe('contactRadius', 4),
        profileType: getStrSafe('profileType', 'depth'),
        fixedValue: getValSafe('fixedValue', 0),
        rangeMin: getValSafe('profileRangeMin', 0),
        rangeMax: getValSafe('profileRangeMax', 30),
        resolution: getValSafe('profileResolution', 0.5),
        selectedResponses: responses.length > 0 ? responses : ['eps_z', 'sigma_z'],
        unitSystem: state.unitSystem
    };
}

// Helper functions for safe element value access
function getValSafe(id, fallback) {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) || fallback : fallback;
}

function getStrSafe(id, fallback) {
    const el = document.getElementById(id);
    return el ? el.value || fallback : fallback;
}

function preparePointwiseData() {
    const responses = [];
    document.querySelectorAll('#pointwiseResponses .checkbox input:checked').forEach(cb => {
        if (cb.value) responses.push(cb.value);
    });

    return {
        layers: state.layers,
        loads: state.loads,
        contactRadius: getValSafe('contactRadius', 4),
        points: getPoints(),
        selectedResponses: responses.length > 0 ? responses : ['eps_z', 'sigma_z', 'deflection_z'],
        unitSystem: state.unitSystem
    };
}

// ============================================================================
// Visualization
// ============================================================================

function renderHeatmap(result) {
    const { x, z, allData, layerBoundaries } = result;

    // Cache the data for instant response switching
    state.cachedHeatmapData = allData;
    state.cachedHeatmapX = x;
    state.cachedHeatmapZ = z;
    state.cachedLayerBoundaries = layerBoundaries;

    // Show the response selector
    document.getElementById('responseSelectorContainer').classList.remove('hidden');

    // Sync the dropdown with current selection
    document.getElementById('displayResponseType').value = state.currentResponse;

    // Render using the cached data
    renderCachedHeatmap();
}

function renderCachedHeatmap() {
    if (!state.cachedHeatmapData) return;

    const container = document.getElementById('plotContainer');
    container.innerHTML = '';

    const respType = state.currentResponse;
    const data = state.cachedHeatmapData[respType];
    const x = state.cachedHeatmapX;
    const z = state.cachedHeatmapZ;
    const layerBoundaries = state.cachedLayerBoundaries;

    if (!data) {
        showToast(`No data for ${respType}`, 'error');
        return;
    }

    // Determine unit label
    let unit = '';
    if (respType.startsWith('eps_')) {
        unit = 'με';
    } else if (respType.startsWith('sigma_')) {
        unit = state.unitSystem === 'imperial' ? 'psi' : 'MPa';
    } else if (respType === 'deflection_z') {
        unit = state.unitSystem === 'imperial' ? 'in' : 'mm';
    }

    // Check interpolation checkbox
    const interpolateEl = document.getElementById('heatmapInterpolate');
    const useInterpolation = interpolateEl ? interpolateEl.checked : true;

    const heatmapTrace = {
        z: data,
        x: x,
        y: z,
        type: 'heatmap',
        colorscale: 'RdBu',
        reversescale: true,
        zmid: 0,
        zsmooth: useInterpolation ? 'best' : false,  // Smooth interpolation or raw pixels
        colorbar: {
            title: { text: unit, side: 'right' },
            tickfont: { color: '#94a3b8', size: 10 },
            titlefont: { color: '#f1f5f9', size: 11 }
        },
        hovertemplate: 'x: %{x}<br>z: %{y}<br>Value: %{z:.4f}<extra></extra>'
    };

    const shapes = layerBoundaries.map(h => ({
        type: 'line',
        x0: Math.min(...x),
        x1: Math.max(...x),
        y0: h,
        y1: h,
        line: { color: 'rgba(255, 255, 255, 0.4)', width: 1.5, dash: 'dash' }
    }));

    const layout = {
        title: {
            text: RESPONSE_NAMES[respType] || respType,
            font: { color: '#f1f5f9', size: 14 }
        },
        xaxis: {
            title: { text: `x (${UNITS[state.unitSystem].length})`, font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.2)',
            automargin: true
        },
        yaxis: {
            title: { text: `z (${UNITS[state.unitSystem].length})`, font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            autorange: 'reversed',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            zerolinecolor: 'rgba(148, 163, 184, 0.2)',
            automargin: true
        },
        shapes: shapes,
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#111827',
        margin: { t: 40, b: 80, l: 70, r: 20 },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot(container, [heatmapTrace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    });

    hideResultsTable();
}

function renderLinePlotFromCache() {
    const container = document.getElementById('plotContainer');
    if (!container) return;

    // Check all required cached data exists
    if (!state.cachedHeatmapData || !state.cachedHeatmapX || !state.cachedHeatmapZ ||
        state.cachedHeatmapX.length === 0 || state.cachedHeatmapZ.length === 0) {
        container.innerHTML = `
            <div class="plot-placeholder">
                <div class="placeholder-icon">📈</div>
                <p>Run analysis first to generate line plots</p>
            </div>
        `;
        return;
    }

    container.innerHTML = '';

    // Get settings from results panel controls (not the old lineplot tab settings)
    const profileTypeEl = document.getElementById('lineplotProfileType');
    const slider = document.getElementById('positionSlider');

    const profileType = profileTypeEl ? profileTypeEl.value : 'depth';
    const sliderIdx = slider ? parseInt(slider.value) || 0 : 0;

    // Get selected responses from line plot checkboxes (in Analysis panel)
    const selectedResponses = [];
    document.querySelectorAll('#lineplotResponses .checkbox input:checked').forEach(cb => {
        if (cb.value) selectedResponses.push(cb.value);
    });

    if (selectedResponses.length === 0) {
        selectedResponses.push('eps_z', 'sigma_z');
    }

    const x = state.cachedHeatmapX;
    const z = state.cachedHeatmapZ;
    const allData = state.cachedHeatmapData;
    const layerBoundaries = state.cachedLayerBoundaries || [];

    // Determine which array to use for profile axis and fixed value
    const isDepth = profileType === 'depth';
    const fixedArray = isDepth ? x : z;  // Array we pick one value from (slider controls this)
    const axisValues = isDepth ? z : x;  // Array that becomes the plot axis

    // The slider selects an index from the fixed array
    const closestIdx = Math.min(sliderIdx, fixedArray.length - 1);
    const fixedValue = fixedArray[closestIdx];

    // Build profiles from cached data
    const profiles = {};

    // Extract profile for each selected response
    for (const respType of selectedResponses) {
        if (allData[respType]) {
            const respData = allData[respType];  // shape: z-rows x x-cols
            const profile = [];
            if (isDepth) {
                // Fixed X (column), vary Z (rows): extract column at closestIdx
                for (let zIdx = 0; zIdx < z.length; zIdx++) {
                    profile.push(respData[zIdx][closestIdx]);
                }
            } else {
                // Fixed Z (row), vary X (columns): extract row at closestIdx
                for (let xIdx = 0; xIdx < x.length; xIdx++) {
                    profile.push(respData[closestIdx][xIdx]);
                }
            }
            profiles[respType] = profile;
        }
    }

    // Build traces with HORIZONTAL orientation: x = position, y = response value
    const colors = ['#818cf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#38bdf8', '#f87171'];
    const traces = [];
    let colorIdx = 0;

    for (const [respType, values] of Object.entries(profiles)) {
        traces.push({
            x: axisValues,  // Position on x-axis (z for depth, x for horizontal)
            y: values,      // Response value on y-axis
            type: 'scatter',
            mode: 'lines',
            name: RESPONSE_NAMES[respType] || respType,
            line: { color: colors[colorIdx % colors.length], width: 2 }
        });
        colorIdx++;
    }

    // Layer boundaries as vertical lines (only for depth profiles)
    const shapes = isDepth ? layerBoundaries.map(h => ({
        type: 'line',
        x0: h,
        x1: h,
        y0: 0,
        y1: 1,
        yref: 'paper',  // Use paper coordinates (0-1) instead of data coordinates
        line: { color: 'rgba(255, 255, 255, 0.25)', width: 1, dash: 'dot' }
    })) : [];

    const axisLabel = isDepth ? 'z' : 'x';
    const fixedLabel = isDepth ? 'X' : 'Z';
    const title = `Profile at ${fixedLabel} = ${fixedValue.toFixed(1)}`;

    const layout = {
        title: { text: title, font: { color: '#f1f5f9', size: 14 } },
        xaxis: {
            title: { text: `${axisLabel} (${UNITS[state.unitSystem].length})`, font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            automargin: true
        },
        yaxis: {
            title: { text: 'Response Value', font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            automargin: true
        },
        shapes: shapes,
        legend: { font: { color: '#94a3b8', size: 10 }, bgcolor: 'rgba(17, 24, 39, 0.8)' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#111827',
        margin: { t: 40, b: 80, l: 70, r: 20 },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot(container, traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    });

    hideResultsTable();
}

function renderProfile(result) {
    const container = document.getElementById('plotContainer');
    container.innerHTML = '';

    const { axisValues, profiles, profileType, fixedValue, layerBoundaries } = result;

    const colors = ['#818cf8', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#38bdf8', '#f87171'];
    const traces = [];
    let colorIdx = 0;

    for (const [respType, values] of Object.entries(profiles)) {
        traces.push({
            x: values,  // Response values always on x-axis
            y: axisValues,  // Position (z or x) always on y-axis
            type: 'scatter',
            mode: 'lines',
            name: RESPONSE_NAMES[respType] || respType,
            line: { color: colors[colorIdx % colors.length], width: 2 }
        });
        colorIdx++;
    }

    const shapes = profileType === 'depth' ? layerBoundaries.map(h => ({
        type: 'line',
        x0: -1e10,
        x1: 1e10,
        y0: h,
        y1: h,
        line: { color: 'rgba(255, 255, 255, 0.25)', width: 1, dash: 'dot' }
    })) : [];

    const xTitle = 'Response Value';
    const yTitle = profileType === 'depth'
        ? `z (${UNITS[state.unitSystem].length})`
        : `x (${UNITS[state.unitSystem].length})`;
    const title = profileType === 'depth'
        ? `Depth Profile at x = ${fixedValue}`
        : `Horizontal Profile at z = ${fixedValue}`;

    const layout = {
        title: { text: title, font: { color: '#f1f5f9', size: 14 } },
        xaxis: {
            title: { text: xTitle, font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            automargin: true
        },
        yaxis: {
            title: { text: yTitle, font: { color: '#94a3b8', size: 11 } },
            tickfont: { color: '#94a3b8', size: 10 },
            autorange: profileType === 'depth' ? 'reversed' : true,
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            automargin: true
        },
        shapes: shapes,
        legend: { font: { color: '#94a3b8', size: 10 }, bgcolor: 'rgba(17, 24, 39, 0.8)' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#111827',
        margin: { t: 40, b: 80, l: 70, r: 20 },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot(container, traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    });

    hideResultsTable();
}

function renderPointwiseResults(result) {
    // Cache the results for when user switches back to pointwise tab
    state.cachedPointwiseData = result;
    renderPointwiseFromCache();
}

function renderPointwiseFromCache() {
    if (!state.cachedPointwiseData) return;

    const { points, responses } = state.cachedPointwiseData;

    // Show placeholder in plot
    document.getElementById('plotContainer').innerHTML = `
        <div class="plot-placeholder">
            <div class="placeholder-icon">📋</div>
            <p>Pointwise results shown in table</p>
        </div>
    `;

    // Build results table
    const table = document.getElementById('resultsTable');
    const thead = table.querySelector('thead tr');
    const tbody = table.querySelector('tbody');

    thead.innerHTML = '<th data-key="x">X</th><th data-key="z">Z</th>';
    responses.forEach(resp => {
        // Use data-key for CSV export (raw LEA name) and show display name visually
        thead.innerHTML += `<th data-key="${resp}">${RESPONSE_NAMES[resp] || resp}</th>`;
    });

    tbody.innerHTML = '';
    points.forEach(point => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${point.x}</td><td>${point.z}</td>`;
        responses.forEach(resp => {
            const val = point[resp];
            tr.innerHTML += `<td>${val !== undefined ? val.toFixed(4) : '-'}</td>`;
        });
        tbody.appendChild(tr);
    });

    showResultsTable();
}

function showResultsTable() {
    document.getElementById('resultsTableContainer').classList.remove('hidden');
    document.getElementById('resultsActions').classList.remove('hidden');
}

function hideResultsTable() {
    document.getElementById('resultsTableContainer').classList.add('hidden');
    document.getElementById('resultsActions').classList.add('hidden');
}

// ============================================================================
// Export Functions
// ============================================================================

function copyResults() {
    const table = document.getElementById('resultsTable');
    let text = '';

    // Headers
    table.querySelectorAll('thead th').forEach(th => {
        text += th.textContent + '\t';
    });
    text = text.trim() + '\n';

    // Rows
    table.querySelectorAll('tbody tr').forEach(tr => {
        tr.querySelectorAll('td').forEach(td => {
            text += td.textContent + '\t';
        });
        text = text.trim() + '\n';
    });

    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

function downloadCSV() {
    const table = document.getElementById('resultsTable');
    let csv = '';

    // Headers - use data-key for raw LEA names, fallback to textContent
    const headers = [];
    table.querySelectorAll('thead th').forEach(th => {
        headers.push(th.dataset.key || th.textContent);
    });
    csv += headers.join(',') + '\n';

    // Rows
    table.querySelectorAll('tbody tr').forEach(tr => {
        const row = [];
        tr.querySelectorAll('td').forEach(td => {
            row.push(td.textContent);
        });
        csv += row.join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'weblea_results.csv';
    a.click();
    URL.revokeObjectURL(url);

    showToast('CSV downloaded', 'success');
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(message, type = 'error') {
    const toast = document.getElementById('toast');
    toast.querySelector('.toast-message').textContent = message;
    toast.classList.remove('hidden', 'success');
    if (type === 'success') {
        toast.classList.add('success');
        toast.querySelector('.toast-icon').textContent = '✓';
    } else {
        toast.querySelector('.toast-icon').textContent = '⚠';
    }

    setTimeout(() => hideToast(), 4000);
}

function hideToast() {
    document.getElementById('toast').classList.add('hidden');
}

// ============================================================================
// Collapsible Panels
// ============================================================================

function togglePanel(headerEl) {
    // Look for .collapsible parent (for input-section-wrapper) or .panel (for individual panels)
    const panel = headerEl.closest('.collapsible') || headerEl.closest('.panel');
    if (panel) {
        panel.classList.toggle('collapsed');
    }
}

// ============================================================================
// Select All Toggle for Pointwise
// ============================================================================

function toggleSelectAll(masterCheckbox) {
    const container = document.getElementById('pointwiseResponses');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]:not(#selectAllPointwise)');
    checkboxes.forEach(cb => {
        cb.checked = masterCheckbox.checked;
    });
}

// ============================================================================
// Info Modal
// ============================================================================

function openInfoModal() {
    document.getElementById('infoModal').classList.add('visible');
}

function closeInfoModal() {
    document.getElementById('infoModal').classList.remove('visible');
}

// Add info button listener after DOM loaded
document.addEventListener('DOMContentLoaded', () => {
    const infoBtn = document.getElementById('infoBtn');
    if (infoBtn) {
        infoBtn.addEventListener('click', openInfoModal);
    }

    // Close modal when clicking overlay
    const modal = document.getElementById('infoModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeInfoModal();
        });
    }
});
