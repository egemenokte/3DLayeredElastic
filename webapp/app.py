"""
WEBLEA - Flask Backend
Layered Elastic Analysis Web Application
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Layer3D from local copy (for cloud deployment compatibility)
from MDA_Huang import Layer3D

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# ============================================================================
# Configuration
# ============================================================================

# Default LEA settings
LEA_CONFIG = {
    'ZRO': 7e-20,           # Zero threshold to avoid division by zero
    'iterations': 1600,     # Maximum iterations
    'tolerance': 0.01,      # Convergence tolerance
    'every': 10             # Check convergence every N steps
}

# Response type metadata
RESPONSE_TYPES = {
    'deflection_z': {'name': 'Vertical Deflection', 'unit': 'length'},
    'eps_x': {'name': 'Normal Strain εx', 'unit': 'microstrain'},
    'eps_y': {'name': 'Normal Strain εy', 'unit': 'microstrain'},
    'eps_z': {'name': 'Normal Strain εz', 'unit': 'microstrain'},
    'eps_xy': {'name': 'Shear Strain γxy', 'unit': 'microstrain'},
    'eps_yz': {'name': 'Shear Strain γyz', 'unit': 'microstrain'},
    'eps_xz': {'name': 'Shear Strain γxz', 'unit': 'microstrain'},
    'sigma_x': {'name': 'Normal Stress σx', 'unit': 'stress'},
    'sigma_y': {'name': 'Normal Stress σy', 'unit': 'stress'},
    'sigma_z': {'name': 'Normal Stress σz', 'unit': 'stress'},
    'sigma_xy': {'name': 'Shear Stress τxy', 'unit': 'stress'},
    'sigma_yz': {'name': 'Shear Stress τyz', 'unit': 'stress'},
    'sigma_xz': {'name': 'Shear Stress τxz', 'unit': 'stress'}
}

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'app': 'WEBLEA'})

@app.route('/api/responses')
def get_responses():
    """Get available response types"""
    return jsonify(RESPONSE_TYPES)

# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.route('/api/analyze', methods=['POST'])
def analyze_heatmap():
    """
    Heatmap analysis endpoint.
    Computes response values over a 2D grid (x, z).
    """
    try:
        data = request.json
        
        # Extract inputs
        layers = data['layers']
        loads = data['loads']
        contact_radius = float(data['contactRadius'])
        x_min = float(data.get('xMin', 0))
        x_max = float(data.get('xMax', 30))
        z_min = float(data.get('zMin', 0))
        z_max = float(data.get('zMax', 30))
        resolution = float(data.get('resolution', 1))
        
        # Prepare layer properties
        E = np.array([layer['modulus'] for layer in layers])
        H = [layer['thickness'] for layer in layers[:-1]]  # Exclude subgrade
        nu = [layer['poisson'] for layer in layers]
        
        # Prepare loads
        unit_system = data.get('unitSystem', 'imperial')
        load_scale = 1000.0 if unit_system == 'si' else 1.0  # SI UI uses kN, solver expects force consistent with MPa-mm (N)
        L = [load['magnitude'] * load_scale for load in loads]
        LPos = [(load['x'], 0) for load in loads]  # y=0 for 2D analysis
        
        # Generate grid
        x_values = np.arange(x_min, x_max + resolution/2, resolution)
        z_values = np.arange(z_min, z_max + resolution/2, resolution)
        
        # Avoid z=0 (surface singularity)
        z_values = np.maximum(z_values, 0.01)
        
        # Add artificial points just before and after layer transitions
        # This captures the discontinuity at layer interfaces
        layer_thicknesses = [layer['thickness'] for layer in layers[:-1] if layer['thickness'] is not None]
        if layer_thicknesses:
            cumulative_depths = np.cumsum(layer_thicknesses)
            # Add points at depth - 0.01 and depth + 0.01 for each layer boundary
            transition_points = np.concatenate([cumulative_depths - 0.01, cumulative_depths + 0.01])
            # Filter to only include points within z range
            transition_points = transition_points[(transition_points >= z_min) & (transition_points <= z_max)]
            z_values = np.sort(np.unique(np.concatenate([z_values, transition_points])))
        
        y_values = [0]  # 2D analysis at y=0
        
        # Run LEA
        results = Layer3D(
            L, LPos, contact_radius,
            x_values.tolist(), y_values, z_values.tolist(),
            H, E, nu,
            LEA_CONFIG['iterations'],
            LEA_CONFIG['ZRO'],
            np.ones(len(E)),  # Fully bonded
            LEA_CONFIG['tolerance'],
            verbose=False,
            every=LEA_CONFIG['every']
        )
        
        # All response types to extract
        all_responses = [
            'deflection_z', 
            'eps_x', 'eps_y', 'eps_z', 'eps_xy', 'eps_yz', 'eps_xz',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xy', 'sigma_yz', 'sigma_xz'
        ]
        
        # Extract ALL response data and convert to proper format
        all_data = {}
        for resp_type in all_responses:
            if resp_type in results:
                response_data = results[resp_type].copy()
                
                # Convert strains to microstrain (×1e6) for better readability
                if resp_type.startswith('eps_'):
                    response_data = response_data * 1e6
                
                # Reshape for heatmap (z rows, x columns)
                all_data[resp_type] = response_data[0, :, :].T.tolist()
        
        # Calculate layer boundaries
        layer_boundaries = []
        cumulative = 0
        for h in H:
            cumulative += h
            if cumulative <= z_max:
                layer_boundaries.append(cumulative)
        
        return jsonify({
            'success': True,
            'x': x_values.tolist(),
            'z': z_values.tolist(),
            'allData': all_data,
            'layerBoundaries': layer_boundaries
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze-profile', methods=['POST'])
def analyze_profile():
    """
    Line profile analysis endpoint.
    Computes response values along a line (depth or horizontal).
    """
    try:
        data = request.json
        
        # Extract inputs
        layers = data['layers']
        loads = data['loads']
        contact_radius = float(data['contactRadius'])
        profile_type = data.get('profileType', 'depth')  # 'depth' or 'horizontal'
        fixed_value = float(data.get('fixedValue', 0))
        range_min = float(data.get('rangeMin', 0))
        range_max = float(data.get('rangeMax', 30))
        resolution = float(data.get('resolution', 0.5))
        selected_responses = data.get('selectedResponses', ['eps_z', 'sigma_z'])
        
        # Prepare layer properties
        E = np.array([layer['modulus'] for layer in layers])
        H = [layer['thickness'] for layer in layers[:-1]]
        nu = [layer['poisson'] for layer in layers]
        
        # Prepare loads
        unit_system = data.get('unitSystem', 'imperial')
        load_scale = 1000.0 if unit_system == 'si' else 1.0  # SI UI uses kN, solver expects force consistent with MPa-mm (N)
        L = [load['magnitude'] * load_scale for load in loads]
        LPos = [(load['x'], 0) for load in loads]
        
        # Generate points based on profile type
        axis_values = np.arange(range_min, range_max + resolution/2, resolution)
        
        if profile_type == 'depth':
            x_values = [fixed_value]
            z_values = np.maximum(axis_values, 0.01).tolist()
        else:
            x_values = axis_values.tolist()
            z_values = [max(fixed_value, 0.01)]
        
        y_values = [0]
        
        # Run LEA
        results = Layer3D(
            L, LPos, contact_radius,
            x_values, y_values, z_values,
            H, E, nu,
            LEA_CONFIG['iterations'],
            LEA_CONFIG['ZRO'],
            np.ones(len(E)),
            LEA_CONFIG['tolerance'],
            verbose=False,
            every=LEA_CONFIG['every']
        )
        
        # Extract profiles
        profiles = {}
        for resp in selected_responses:
            if resp in results:
                resp_data = results[resp]
                # Convert strains to microstrain (×1e6)
                if resp.startswith('eps_'):
                    resp_data = resp_data * 1e6
                if profile_type == 'depth':
                    # Shape: (y, x, z) -> extract (0, 0, :) for depth profile
                    profiles[resp] = resp_data[0, 0, :].tolist()
                else:
                    # Shape: (y, x, z) -> extract (0, :, 0) for horizontal profile
                    profiles[resp] = resp_data[0, :, 0].tolist()
        
        # Layer boundaries
        layer_boundaries = []
        cumulative = 0
        for h in H:
            cumulative += h
            if cumulative <= range_max:
                layer_boundaries.append(cumulative)
        
        return jsonify({
            'success': True,
            'axisValues': axis_values.tolist(),
            'profiles': profiles,
            'profileType': profile_type,
            'fixedValue': fixed_value,
            'layerBoundaries': layer_boundaries
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analyze-points', methods=['POST'])
def analyze_points():
    """
    Pointwise analysis endpoint.
    Computes response values at specific (x, z) points.
    """
    try:
        data = request.json
        
        # Extract inputs
        layers = data['layers']
        loads = data['loads']
        contact_radius = float(data['contactRadius'])
        points = data.get('points', [])
        selected_responses = data.get('selectedResponses', ['eps_z', 'sigma_z', 'deflection_z'])
        
        if not points:
            return jsonify({'error': 'No points specified'}), 400
        
        # Prepare layer properties
        E = np.array([layer['modulus'] for layer in layers])
        H = [layer['thickness'] for layer in layers[:-1]]
        nu = [layer['poisson'] for layer in layers]
        
        # Prepare loads
        unit_system = data.get('unitSystem', 'imperial')
        load_scale = 1000.0 if unit_system == 'si' else 1.0  # SI UI uses kN, solver expects force consistent with MPa-mm (N)
        L = [load['magnitude'] * load_scale for load in loads]
        LPos = [(load['x'], 0) for load in loads]
        
        # Extract unique x and z values
        x_values = sorted(list(set(p['x'] for p in points)))
        z_values = sorted(list(set(max(p['z'], 0.01) for p in points)))
        y_values = [0]
        
        # Run LEA
        results = Layer3D(
            L, LPos, contact_radius,
            x_values, y_values, z_values,
            H, E, nu,
            LEA_CONFIG['iterations'],
            LEA_CONFIG['ZRO'],
            np.ones(len(E)),
            LEA_CONFIG['tolerance'],
            verbose=False,
            every=LEA_CONFIG['every']
        )
        
        # Build output for each point
        output_points = []
        for p in points:
            x_idx = x_values.index(p['x'])
            z_val = max(p['z'], 0.01)
            z_idx = z_values.index(z_val)
            
            point_data = {'x': p['x'], 'z': p['z']}
            for resp in selected_responses:
                if resp in results:
                    val = float(results[resp][0, x_idx, z_idx])
                    # Convert strains to microstrain (×1e6)
                    if resp.startswith('eps_'):
                        val = val * 1e6
                    point_data[resp] = val
            
            output_points.append(point_data)
        
        return jsonify({
            'success': True,
            'points': output_points,
            'responses': selected_responses
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("  WEBLEA - Layered Elastic Analysis")
    print("  Starting server at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
