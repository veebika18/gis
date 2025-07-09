import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import tempfile
import os
import folium
mymap = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
mymap.save("map.html")

import re
from scipy import ndimage
from rasterio.transform import from_bounds
import rasterio
from streamlit_folium import folium_static


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPQueryProcessor:
    """Natural Language Processing for location extraction"""
    
    def __init__(self):
        self.city_patterns = {
            'chennai': [
                r'\b(chennai|madras|tamil nadu capital|tamilnadu capital)\b',
                r'\bchennai\b',
                r'\bmadras\b',
                r'\btamil nadu\b.*\bcity\b',
                r'\bsouth india\b.*\bcoastal\b.*\bcity\b'
            ],
            'mumbai': [
                r'\b(mumbai|bombay|maharashtra capital|financial capital)\b',
                r'\bmumbai\b',
                r'\bbombay\b',
                r'\bmaharashtra\b.*\bcity\b',
                r'\bfinancial capital\b.*\bindia\b'
            ],
            'bangalore': [
                r'\b(bangalore|bengaluru|karnataka capital|silicon valley)\b',
                r'\bbangalore\b',
                r'\bbengaluru\b',
                r'\bkarnataka\b.*\bcity\b',
                r'\bsilicon valley\b.*\bindia\b',
                r'\bit capital\b.*\bindia\b'
            ],
            'delhi': [
                r'\b(delhi|new delhi|national capital|capital of india)\b',
                r'\bdelhi\b',
                r'\bnew delhi\b',
                r'\bcapital\b.*\bindia\b',
                r'\bnational capital\b'
            ]
        }
        
        self.flood_keywords = [
            'flood', 'flooding', 'water logging', 'inundation', 'deluge',
            'monsoon', 'rainfall', 'drainage', 'waterlogging', 'submersion'
        ]
        
        self.risk_keywords = [
            'risk', 'assessment', 'analysis', 'vulnerability', 'hazard',
            'evaluation', 'mapping', 'prediction', 'forecasting'
        ]
    
    def extract_city_from_query(self, query: str) -> str:
        """Extract city name from natural language query"""
        query_lower = query.lower()
        
        # Check for direct city mentions
        for city, patterns in self.city_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return city.title()
        
        # If no city found, return default
        return "Chennai"
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent and parameters from query"""
        query_lower = query.lower()
        
        intent = {
            'city': self.extract_city_from_query(query),
            'is_flood_related': any(keyword in query_lower for keyword in self.flood_keywords),
            'is_risk_analysis': any(keyword in query_lower for keyword in self.risk_keywords),
            'urgency': 'high' if any(word in query_lower for word in ['urgent', 'immediate', 'emergency']) else 'normal',
            'time_frame': 'current' if any(word in query_lower for word in ['current', 'now', 'today']) else 'general'
        }
        
        return intent
    
    def generate_response_context(self, intent: Dict[str, Any]) -> str:
        """Generate contextual response based on intent"""
        city = intent['city']
        
        if intent['is_flood_related'] and intent['is_risk_analysis']:
            return f"Analyzing flood risk for {city}. This comprehensive assessment will evaluate elevation, drainage, water proximity, and other critical factors."
        elif intent['is_flood_related']:
            return f"Flood analysis requested for {city}. Generating detailed flood risk assessment with interactive visualizations."
        elif intent['is_risk_analysis']:
            return f"Risk assessment for {city}. Evaluating multiple risk factors including topography, land use, and infrastructure."
        else:
            return f"Geographic analysis for {city}. Performing comprehensive flood risk evaluation."

class WorkflowLogger:
    """Logger for tracking analysis workflow steps"""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.logs = []
    
    def add_step(self, step_name: str, description: str):
        """Add a workflow step"""
        self.steps.append({
            'step': len(self.steps) + 1,
            'name': step_name,
            'description': description,
            'status': 'pending',
            'timestamp': None,
            'duration': None
        })
    
    def start_step(self, step_index: int):
        """Start a workflow step"""
        if step_index < len(self.steps):
            self.steps[step_index]['status'] = 'running'
            self.steps[step_index]['timestamp'] = datetime.now()
            self.current_step = step_index
    
    def complete_step(self, step_index: int, details: str = None):
        """Complete a workflow step"""
        if step_index < len(self.steps):
            step = self.steps[step_index]
            step['status'] = 'completed'
            if step['timestamp']:
                step['duration'] = (datetime.now() - step['timestamp']).total_seconds()
            if details:
                step['details'] = details
            self.logs.append(f"✅ {step['name']}: {details or 'Completed'}")
    
    def get_workflow_json(self) -> str:
        """Get workflow as JSON"""
        return json.dumps(self.steps, indent=2, default=str)
    
    def get_progress(self) -> float:
        """Get current progress percentage"""
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        return (completed / len(self.steps)) * 100 if self.steps else 0

class EnhancedLocationManager:
    """Enhanced location manager with detailed geographic data"""
    
    def __init__(self):
        self.location_data = {
            'chennai': {
                'country': 'India',
                'state': 'Tamil Nadu',
                'bounds': (80.0, 12.8, 80.35, 13.25),
                'center': (80.175, 13.025),
                'elevation_range': (0, 60),
                'rainfall_pattern': 'monsoon_heavy',
                'coastal': True,
                'river_systems': ['Cooum', 'Adyar', 'Kosasthalaiyar'],
                'areas': [
                    {"name": "Velachery", "coords": (80.22, 12.98), "type": "residential", "elevation": 25, "risk_factors": ["low_lying", "poor_drainage"]},
                    {"name": "Adyar", "coords": (80.26, 13.00), "type": "residential", "elevation": 15, "risk_factors": ["river_proximity", "coastal"]},
                    {"name": "T. Nagar", "coords": (80.23, 13.04), "type": "commercial", "elevation": 20, "risk_factors": ["high_density", "concrete_surface"]},
                    {"name": "Anna Nagar", "coords": (80.21, 13.08), "type": "residential", "elevation": 35, "risk_factors": ["moderate_elevation"]},
                    {"name": "Guindy", "coords": (80.22, 13.01), "type": "industrial", "elevation": 30, "risk_factors": ["industrial_runoff"]},
                    {"name": "Mylapore", "coords": (80.27, 13.03), "type": "cultural", "elevation": 18, "risk_factors": ["coastal", "heritage_area"]},
                    {"name": "Porur", "coords": (80.16, 13.03), "type": "residential", "elevation": 40, "risk_factors": ["lake_proximity"]},
                    {"name": "Tambaram", "coords": (80.12, 12.92), "type": "suburban", "elevation": 22, "risk_factors": ["suburban_flooding"]},
                    {"name": "Sholinganallur", "coords": (80.23, 12.90), "type": "it_hub", "elevation": 12, "risk_factors": ["very_low_lying", "poor_drainage"]},
                    {"name": "OMR", "coords": (80.24, 12.93), "type": "it_corridor", "elevation": 18, "risk_factors": ["coastal", "rapid_development"]}
                ],
                'water_bodies': [
                    {"name": "Cooum River", "type": "river", "coords": [(80.12, 13.05), (80.28, 13.08)], "width": 50, "depth": 2},
                    {"name": "Adyar River", "type": "river", "coords": [(80.15, 12.98), (80.26, 13.00)], "width": 80, "depth": 3},
                    {"name": "Bay of Bengal", "type": "sea", "coords": [(80.25, 12.85), (80.35, 13.20)], "width": 1000, "depth": 50},
                    {"name": "Chembarambakkam Lake", "type": "lake", "coords": [(80.08, 12.98)], "width": 200, "depth": 5},
                    {"name": "Buckingham Canal", "type": "canal", "coords": [(80.20, 12.90), (80.25, 13.10)], "width": 30, "depth": 2}
                ]
            },
            'mumbai': {
                'country': 'India',
                'state': 'Maharashtra',
                'bounds': (72.75, 18.85, 73.05, 19.35),
                'center': (72.9, 19.1),
                'elevation_range': (0, 180),
                'rainfall_pattern': 'monsoon_extreme',
                'coastal': True,
                'river_systems': ['Mithi', 'Mahim Creek', 'Thane Creek'],
                'areas': [
                    {"name": "Bandra", "coords": (72.84, 19.06), "type": "residential", "elevation": 45, "risk_factors": ["tidal_influence"]},
                    {"name": "Andheri", "coords": (72.85, 19.12), "type": "residential", "elevation": 35, "risk_factors": ["airport_proximity", "drainage_issues"]},
                    {"name": "Powai", "coords": (72.91, 19.12), "type": "residential", "elevation": 85, "risk_factors": ["lake_proximity", "hills"]},
                    {"name": "Colaba", "coords": (72.81, 18.92), "type": "commercial", "elevation": 8, "risk_factors": ["coastal", "low_lying"]},
                    {"name": "Worli", "coords": (72.82, 19.01), "type": "commercial", "elevation": 12, "risk_factors": ["coastal", "reclaimed_land"]},
                    {"name": "Juhu", "coords": (72.83, 19.10), "type": "residential", "elevation": 15, "risk_factors": ["coastal", "airport_runway"]},
                    {"name": "Malad", "coords": (72.84, 19.18), "type": "residential", "elevation": 55, "risk_factors": ["creek_proximity"]},
                    {"name": "Kurla", "coords": (72.88, 19.07), "type": "residential", "elevation": 25, "risk_factors": ["mithi_river", "slums"]}
                ],
                'water_bodies': [
                    {"name": "Mithi River", "type": "river", "coords": [(72.85, 19.05), (72.90, 19.08)], "width": 40, "depth": 2},
                    {"name": "Arabian Sea", "type": "sea", "coords": [(72.75, 18.85), (72.85, 19.35)], "width": 1000, "depth": 100},
                    {"name": "Mahim Creek", "type": "creek", "coords": [(72.83, 19.04), (72.85, 19.06)], "width": 60, "depth": 3},
                    {"name": "Powai Lake", "type": "lake", "coords": [(72.91, 19.12)], "width": 150, "depth": 8},
                    {"name": "Thane Creek", "type": "creek", "coords": [(72.95, 19.15), (72.98, 19.20)], "width": 80, "depth": 4}
                ]
            },
            'bangalore': {
                'country': 'India',
                'state': 'Karnataka',
                'bounds': (77.45, 12.85, 77.75, 13.15),
                'center': (77.6, 13.0),
                'elevation_range': (800, 950),
                'rainfall_pattern': 'moderate_seasonal',
                'coastal': False,
                'river_systems': ['Vrishabhavathi', 'Arkavathy', 'Dakshina Pinakini'],
                'areas': [
                    {"name": "Koramangala", "coords": (77.63, 12.93), "type": "residential", "elevation": 920, "risk_factors": ["urban_flooding"]},
                    {"name": "Whitefield", "coords": (77.75, 12.97), "type": "it_hub", "elevation": 890, "risk_factors": ["rapid_development"]},
                    {"name": "Electronic City", "coords": (77.66, 12.84), "type": "it_hub", "elevation": 870, "risk_factors": ["industrial_area"]},
                    {"name": "Hebbal", "coords": (77.60, 13.04), "type": "residential", "elevation": 930, "risk_factors": ["lake_proximity"]},
                    {"name": "Bellandur", "coords": (77.68, 12.93), "type": "residential", "elevation": 885, "risk_factors": ["lake_pollution", "low_lying"]}
                ],
                'water_bodies': [
                    {"name": "Vrishabhavathi River", "type": "river", "coords": [(77.50, 12.90), (77.65, 13.00)], "width": 25, "depth": 1.5},
                    {"name": "Bellandur Lake", "type": "lake", "coords": [(77.68, 12.93)], "width": 300, "depth": 4},
                    {"name": "Hebbal Lake", "type": "lake", "coords": [(77.60, 13.04)], "width": 180, "depth": 3}
                ]
            },
            'delhi': {
                'country': 'India',
                'state': 'Delhi',
                'bounds': (77.0, 28.4, 77.35, 28.9),
                'center': (77.175, 28.65),
                'elevation_range': (200, 280),
                'rainfall_pattern': 'monsoon_moderate',
                'coastal': False,
                'river_systems': ['Yamuna', 'Najafgarh'],
                'areas': [
                    {"name": "Connaught Place", "coords": (77.22, 28.63), "type": "commercial", "elevation": 216, "risk_factors": ["central_delhi"]},
                    {"name": "Dwarka", "coords": (77.05, 28.58), "type": "residential", "elevation": 225, "risk_factors": ["planned_city"]},
                    {"name": "Gurgaon", "coords": (77.03, 28.46), "type": "commercial", "elevation": 230, "risk_factors": ["rapid_development"]},
                    {"name": "Yamuna Bank", "coords": (77.28, 28.67), "type": "residential", "elevation": 210, "risk_factors": ["river_proximity", "flood_plains"]}
                ],
                'water_bodies': [
                    {"name": "Yamuna River", "type": "river", "coords": [(77.25, 28.40), (77.30, 28.80)], "width": 200, "depth": 5},
                    {"name": "Najafgarh Drain", "type": "drain", "coords": [(77.00, 28.50), (77.15, 28.65)], "width": 15, "depth": 2}
                ]
            }
        }
    
    def get_location_info(self, location_name: str) -> Dict[str, Any]:
        """Get enhanced location information"""
        location_key = location_name.lower().replace(' ', '_')
        return self.location_data.get(location_key, self.location_data['chennai'])

class AdvancedFloodRiskAnalyzer:
    """Advanced flood risk analyzer with comprehensive risk assessment"""
    
    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
        self.location_manager = EnhancedLocationManager()
        self.risk_factors = {
            'elevation_weight': 0.35,
            'slope_weight': 0.25,
            'water_proximity_weight': 0.20,
            'drainage_weight': 0.15,
            'land_use_weight': 0.05
        }
        self.risk_categories = {
            'very_low': {'range': (0, 0.2), 'color': '#00ff00', 'label': 'Very Low'},
            'low': {'range': (0.2, 0.4), 'color': '#ffff00', 'label': 'Low'},
            'moderate': {'range': (0.4, 0.6), 'color': '#ffa500', 'label': 'Moderate'},
            'high': {'range': (0.6, 0.8), 'color': '#ff4500', 'label': 'High'},
            'very_high': {'range': (0.8, 1.0), 'color': '#ff0000', 'label': 'Very High'}
        }
    
    def generate_realistic_elevation_data(self, location_info: Dict[str, Any]) -> np.ndarray:
        """Generate realistic elevation data based on location characteristics"""
        self.logger.start_step(0)
        
        bounds = location_info['bounds']
        elevation_range = location_info['elevation_range']
        
        width, height = 400, 400  # Reduced size for better performance
        x = np.linspace(bounds[0], bounds[2], width)
        y = np.linspace(bounds[1], bounds[3], height)
        X, Y = np.meshgrid(x, y)
        
        # Base elevation
        base_elevation = elevation_range[0]
        elevation_span = elevation_range[1] - elevation_range[0]
        
        # Create realistic topography
        elevation = np.zeros((height, width))
        
        # Add multiple elevation features
        np.random.seed(42)  # For reproducible results
        for i in range(3):
            hill_x = bounds[0] + (bounds[2] - bounds[0]) * np.random.random()
            hill_y = bounds[1] + (bounds[3] - bounds[1]) * np.random.random()
            hill_height = elevation_span * (0.3 + 0.4 * np.random.random())
            hill_spread = 0.02 + 0.01 * np.random.random()
            
            elevation += hill_height * np.exp(-((X - hill_x)**2 + (Y - hill_y)**2) / hill_spread)
        
        # Add coastal slope if coastal city
        if location_info.get('coastal', False):
            coastal_gradient = np.linspace(1, 0, width)
            elevation += elevation_span * 0.3 * coastal_gradient[np.newaxis, :]
        
        # Add river valleys
        for water_body in location_info.get('water_bodies', []):
            if water_body['type'] == 'river':
                coords = water_body['coords']
                if len(coords) >= 2:
                    for coord in coords:
                        river_x, river_y = coord
                        river_depression = 20 * np.exp(-((X - river_x)**2 + (Y - river_y)**2) / 0.001)
                        elevation -= river_depression
        
        elevation += base_elevation
        elevation = np.maximum(elevation, base_elevation)
        
        # Add noise for realism
        noise = np.random.normal(0, elevation_span * 0.02, (height, width))
        elevation += noise
        
        self.logger.complete_step(0, f"Generated {width}x{height} elevation grid")
        return elevation.astype(np.float32)
    
    def calculate_slope_advanced(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate slope with advanced gradient analysis"""
        self.logger.start_step(1)
        
        grad_y, grad_x = np.gradient(elevation)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        slope_degrees = np.arctan(slope_magnitude) * 180 / np.pi
        slope_smoothed = ndimage.gaussian_filter(slope_degrees, sigma=1.0)
        
        self.logger.complete_step(1, f"Calculated slope grid with max slope {np.max(slope_smoothed):.2f}°")
        return slope_smoothed
    
    def calculate_water_distance_advanced(self, location_info: Dict[str, Any], 
                                        elevation_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate advanced water proximity"""
        self.logger.start_step(2)
        
        bounds = location_info['bounds']
        height, width = elevation_shape
        
        x_coords = np.linspace(bounds[0], bounds[2], width)
        y_coords = np.linspace(bounds[1], bounds[3], height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        water_distance = np.full((height, width), np.inf)
        
        water_bodies_processed = 0
        for water_body in location_info.get('water_bodies', []):
            water_type = water_body['type']
            coords = water_body['coords']
            
            if water_type == 'river':
                if len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        start = coords[i]
                        end = coords[i + 1]
                        
                        river_points = []
                        for t in np.linspace(0, 1, 50):
                            river_x = start[0] + t * (end[0] - start[0])
                            river_y = start[1] + t * (end[1] - start[1])
                            river_points.append((river_x, river_y))
                        
                        for river_point in river_points:
                            dist = np.sqrt((X - river_point[0])**2 + (Y - river_point[1])**2)
                            water_distance = np.minimum(water_distance, dist)
            
            elif water_type in ['lake', 'pond']:
                center = coords[0]
                radius = water_body.get('width', 100) / 111000
                
                dist_to_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                water_distance = np.minimum(water_distance, np.maximum(0, dist_to_center - radius))
            
            elif water_type in ['sea', 'bay']:
                if location_info.get('coastal', False):
                    coastal_distance = np.abs(X - bounds[2]) * 0.5
                    water_distance = np.minimum(water_distance, coastal_distance)
            
            water_bodies_processed += 1
        
        water_distance_meters = water_distance * 111000
        
        self.logger.complete_step(2, f"Processed {water_bodies_processed} water bodies")
        return water_distance_meters
    
    def calculate_drainage_capacity(self, location_info: Dict[str, Any], 
                                  elevation_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate drainage capacity"""
        self.logger.start_step(3)
        
        height, width = elevation_shape
        drainage = np.ones((height, width)) * 0.5
        
        bounds = location_info['bounds']
        x_coords = np.linspace(bounds[0], bounds[2], width)
        y_coords = np.linspace(bounds[1], bounds[3], height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        areas_processed = 0
        for area in location_info.get('areas', []):
            area_x, area_y = area['coords']
            
            influence_radius = 0.02
            distance_to_area = np.sqrt((X - area_x)**2 + (Y - area_y)**2)
            influence_mask = distance_to_area < influence_radius
            
            if area['type'] == 'commercial':
                drainage[influence_mask] *= 0.7
            elif area['type'] == 'industrial':
                drainage[influence_mask] *= 0.6
            elif area['type'] == 'residential':
                drainage[influence_mask] *= 0.8
            
            risk_factors = area.get('risk_factors', [])
            if 'poor_drainage' in risk_factors:
                drainage[influence_mask] *= 0.4
            if 'concrete_surface' in risk_factors:
                drainage[influence_mask] *= 0.5
            
            areas_processed += 1
        
        self.logger.complete_step(3, f"Processed drainage for {areas_processed} areas")
        return drainage
    
    def calculate_land_use_risk(self, location_info: Dict[str, Any], 
                               elevation_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate land use risk factor"""
        self.logger.start_step(4)
        
        height, width = elevation_shape
        bounds = location_info['bounds']
        
        land_use_risk = np.ones((height, width)) * 0.3
        
        x_coords = np.linspace(bounds[0], bounds[2], width)
        y_coords = np.linspace(bounds[1], bounds[3], height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        for area in location_info.get('areas', []):
            area_x, area_y = area['coords']
            
            influence_radius = 0.025
            distance_to_area = np.sqrt((X - area_x)**2 + (Y - area_y)**2)
            influence_mask = distance_to_area < influence_radius
            
            area_type = area['type']
            if area_type == 'commercial':
                land_use_risk[influence_mask] = 0.7
            elif area_type == 'industrial':
                land_use_risk[influence_mask] = 0.6
            elif area_type == 'residential':
                land_use_risk[influence_mask] = 0.4
            elif area_type == 'it_hub':
                land_use_risk[influence_mask] = 0.5
        
        self.logger.complete_step(4, "Calculated land use risk patterns")
        return land_use_risk
    
    def calculate_comprehensive_flood_risk(self, location_name: str) -> Dict[str, Any]:
        """Calculate comprehensive flood risk analysis"""
        self.logger.start_step(5)
        
        location_info = self.location_manager.get_location_info(location_name)
        
        elevation = self.generate_realistic_elevation_data(location_info)
        slope = self.calculate_slope_advanced(elevation)
        water_distance = self.calculate_water_distance_advanced(location_info, elevation.shape)
        drainage = self.calculate_drainage_capacity(location_info, elevation.shape)
        land_use_risk = self.calculate_land_use_risk(location_info, elevation.shape)
        
        # Normalize factors
        elevation_norm = self._normalize_elevation_risk(elevation, location_info)
        slope_norm = self._normalize_slope_risk(slope)
        water_norm = self._normalize_water_risk(water_distance)
        drainage_norm = 1 - drainage
        
        # Calculate composite flood risk
        flood_risk = (
            self.risk_factors['elevation_weight'] * elevation_norm +
            self.risk_factors['slope_weight'] * slope_norm +
            self.risk_factors['water_proximity_weight'] * water_norm +
            self.risk_factors['drainage_weight'] * drainage_norm +
            self.risk_factors['land_use_weight'] * land_use_risk
        )
        
        risk_classified = self._classify_risk_levels(flood_risk)
        
        bounds = location_info['bounds']
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                              elevation.shape[1], elevation.shape[0])
        
        self.logger.complete_step(5, f"Completed comprehensive risk analysis for {location_name}")
        
        return {
            'location_info': location_info,
            'elevation': elevation,
            'slope': slope,
            'water_distance': water_distance,
            'drainage': drainage,
            'land_use_risk': land_use_risk,
            'flood_risk': flood_risk,
            'risk_classified': risk_classified,
            'transform': transform,
            'bounds': bounds
        }
    
    def _normalize_elevation_risk(self, elevation: np.ndarray, location_info: Dict) -> np.ndarray:
        """Normalize elevation to risk factor"""
        elevation_range = location_info['elevation_range']
        normalized = (elevation - elevation_range[0]) / (elevation_range[1] - elevation_range[0])
        risk = 1 - normalized
        return np.clip(risk, 0, 1)
    
    def _normalize_slope_risk(self, slope: np.ndarray) -> np.ndarray:
        """Normalize slope to risk factor"""
        slope_max = np.percentile(slope, 95)
        normalized = slope / slope_max
        risk = 1 - normalized
        return np.clip(risk, 0, 1)
    
    def _normalize_water_risk(self, water_distance: np.ndarray) -> np.ndarray:
        """Normalize water distance to risk factor"""
        # Closer to water = higher risk
        max_distance = np.percentile(water_distance, 95)
        normalized = water_distance / max_distance
        risk = 1 - normalized
        return np.clip(risk, 0, 1)
    
    def _classify_risk_levels(self, flood_risk: np.ndarray) -> np.ndarray:
        """Classify risk levels into categories"""
        risk_classified = np.zeros_like(flood_risk, dtype=int)
        
        for i, (category, info) in enumerate(self.risk_categories.items()):
            min_risk, max_risk = info['range']
            mask = (flood_risk >= min_risk) & (flood_risk < max_risk)
            risk_classified[mask] = i
        
        return risk_classified

class AdvancedDataVisualizer:
    """Advanced visualization system for flood risk data"""
    
    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
        self.risk_analyzer = AdvancedFloodRiskAnalyzer(logger)
    
    def create_elevation_map(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create detailed elevation map"""
        self.logger.start_step(6)
        
        elevation = analysis_results['elevation']
        bounds = analysis_results['bounds']
        location_info = analysis_results['location_info']
        
        # Create coordinate arrays
        x_coords = np.linspace(bounds[0], bounds[2], elevation.shape[1])
        y_coords = np.linspace(bounds[1], bounds[3], elevation.shape[0])
        
        fig = go.Figure()
        
        # Add elevation contour
        fig.add_trace(go.Contour(
            z=elevation,
            x=x_coords,
            y=y_coords,
            colorscale='earth',
            name='Elevation',
            showscale=True,
            colorbar=dict(title="Elevation (m)", x=1.02)
        ))
        
        # Add area markers
        for area in location_info.get('areas', []):
            fig.add_trace(go.Scatter(
                x=[area['coords'][0]],
                y=[area['coords'][1]],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=area['name'],
                textposition="top center",
                name=area['name'],
                hovertemplate=f"<b>{area['name']}</b><br>" +
                            f"Type: {area['type']}<br>" +
                            f"Elevation: {area['elevation']}m<br>" +
                            f"Risk Factors: {', '.join(area.get('risk_factors', []))}<extra></extra>"
            ))
        
        # Add water bodies
        for water_body in location_info.get('water_bodies', []):
            if water_body['type'] == 'river':
                coords = water_body['coords']
                if len(coords) >= 2:
                    x_water = [coord[0] for coord in coords]
                    y_water = [coord[1] for coord in coords]
                    fig.add_trace(go.Scatter(
                        x=x_water,
                        y=y_water,
                        mode='lines',
                        line=dict(color='blue', width=3),
                        name=water_body['name'],
                        hovertemplate=f"<b>{water_body['name']}</b><br>" +
                                    f"Type: {water_body['type']}<br>" +
                                    f"Width: {water_body.get('width', 'N/A')}m<extra></extra>"
                    ))
        
        fig.update_layout(
            title=f"Elevation Map - {location_info['state']}, {location_info['country']}",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            showlegend=True,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        self.logger.complete_step(6, "Created elevation visualization")
        return fig
    
    def create_flood_risk_heatmap(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive flood risk heatmap"""
        self.logger.start_step(7)
        
        flood_risk = analysis_results['flood_risk']
        bounds = analysis_results['bounds']
        location_info = analysis_results['location_info']
        
        x_coords = np.linspace(bounds[0], bounds[2], flood_risk.shape[1])
        y_coords = np.linspace(bounds[1], bounds[3], flood_risk.shape[0])
        
        fig = go.Figure()
        
        # Add flood risk heatmap
        fig.add_trace(go.Heatmap(
            z=flood_risk,
            x=x_coords,
            y=y_coords,
            colorscale=[
                [0, '#00ff00'],    # Very Low - Green
                [0.25, '#ffff00'], # Low - Yellow
                [0.5, '#ffa500'],  # Moderate - Orange
                [0.75, '#ff4500'], # High - Red Orange
                [1, '#ff0000']     # Very High - Red
            ],
            name='Flood Risk',
            colorbar=dict(
                title="Flood Risk Level",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            )
        ))
        
        # Add critical area annotations
        for area in location_info.get('areas', []):
            risk_factors = area.get('risk_factors', [])
            if any(factor in risk_factors for factor in ['poor_drainage', 'very_low_lying', 'flood_prone']):
                fig.add_annotation(
                    x=area['coords'][0],
                    y=area['coords'][1],
                    text=f"⚠️ {area['name']}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    bgcolor="yellow",
                    bordercolor="red",
                    borderwidth=2
                )
        
        fig.update_layout(
            title=f"Comprehensive Flood Risk Assessment - {location_info['state']}",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        self.logger.complete_step(7, "Created flood risk heatmap")
        return fig
    
    def create_risk_factor_analysis(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create multi-factor risk analysis"""
        self.logger.start_step(8)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Elevation Risk', 'Slope Risk', 'Water Proximity Risk', 
                          'Drainage Capacity', 'Land Use Risk', 'Composite Risk'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        bounds = analysis_results['bounds']
        x_coords = np.linspace(bounds[0], bounds[2], analysis_results['elevation'].shape[1])
        y_coords = np.linspace(bounds[1], bounds[3], analysis_results['elevation'].shape[0])
        
        # Elevation risk
        elevation_risk = self.risk_analyzer._normalize_elevation_risk(
            analysis_results['elevation'], analysis_results['location_info'])
        fig.add_trace(go.Heatmap(z=elevation_risk, x=x_coords, y=y_coords, 
                                colorscale='Reds', showscale=False), row=1, col=1)
        
        # Slope risk
        slope_risk = self.risk_analyzer._normalize_slope_risk(analysis_results['slope'])
        fig.add_trace(go.Heatmap(z=slope_risk, x=x_coords, y=y_coords, 
                                colorscale='Oranges', showscale=False), row=1, col=2)
        
        # Water proximity risk
        water_risk = self.risk_analyzer._normalize_water_risk(analysis_results['water_distance'])
        fig.add_trace(go.Heatmap(z=water_risk, x=x_coords, y=y_coords, 
                                colorscale='Blues', showscale=False), row=1, col=3)
        
        # Drainage capacity (inverted for risk)
        drainage_risk = 1 - analysis_results['drainage']
        fig.add_trace(go.Heatmap(z=drainage_risk, x=x_coords, y=y_coords, 
                                colorscale='Purples', showscale=False), row=2, col=1)
        
        # Land use risk
        fig.add_trace(go.Heatmap(z=analysis_results['land_use_risk'], x=x_coords, y=y_coords, 
                                colorscale='Greens', showscale=False), row=2, col=2)
        
        # Composite risk
        fig.add_trace(go.Heatmap(z=analysis_results['flood_risk'], x=x_coords, y=y_coords, 
                                colorscale='RdYlBu_r', showscale=True), row=2, col=3)
        
        fig.update_layout(
            title="Multi-Factor Flood Risk Analysis",
            height=800,
            showlegend=False
        )
        
        self.logger.complete_step(8, "Created multi-factor analysis")
        return fig
    
    def create_area_risk_dashboard(self, analysis_results: Dict[str, Any]) -> go.Figure:
        """Create area-wise risk dashboard"""
        self.logger.start_step(9)
        
        location_info = analysis_results['location_info']
        areas = location_info.get('areas', [])
        
        # Extract risk scores for each area
        area_names = []
        risk_scores = []
        elevations = []
        risk_factors_count = []
        
        for area in areas:
            area_names.append(area['name'])
            
            # Calculate risk score based on factors
            base_risk = 0.3
            if area['elevation'] < 20:
                base_risk += 0.3
            elif area['elevation'] < 50:
                base_risk += 0.2
            
            risk_factors = area.get('risk_factors', [])
            risk_factor_weights = {
                'very_low_lying': 0.4,
                'poor_drainage': 0.3,
                'coastal': 0.2,
                'river_proximity': 0.25,
                'flood_prone': 0.35,
                'low_lying': 0.25,
                'concrete_surface': 0.15,
                'high_density': 0.1,
                'industrial_runoff': 0.1
            }
            
            for factor in risk_factors:
                base_risk += risk_factor_weights.get(factor, 0.1)
            
            risk_scores.append(min(base_risk, 1.0))
            elevations.append(area['elevation'])
            risk_factors_count.append(len(risk_factors))
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Area Risk Scores', 'Elevation vs Risk', 'Risk Factors Distribution', 'Risk Categories'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Risk scores bar chart
        colors = ['red' if score > 0.7 else 'orange' if score > 0.5 else 'yellow' if score > 0.3 else 'green' 
                 for score in risk_scores]
        
        fig.add_trace(go.Bar(
            x=area_names,
            y=risk_scores,
            marker_color=colors,
            name='Risk Score',
            text=[f'{score:.2f}' for score in risk_scores],
            textposition='auto'
        ), row=1, col=1)
        
        # Elevation vs Risk scatter
        fig.add_trace(go.Scatter(
            x=elevations,
            y=risk_scores,
            mode='markers+text',
            text=area_names,
            textposition='top center',
            marker=dict(size=10, color=colors),
            name='Areas'
        ), row=1, col=2)
        
        # Risk factors histogram
        fig.add_trace(go.Histogram(
            x=risk_factors_count,
            nbinsx=max(risk_factors_count) + 1,
            name='Risk Factors'
        ), row=2, col=1)
        
        # Risk categories pie chart
        risk_categories = ['Very High' if score > 0.8 else 'High' if score > 0.6 else 
                          'Moderate' if score > 0.4 else 'Low' for score in risk_scores]
        category_counts = pd.Series(risk_categories).value_counts()
        
        fig.add_trace(go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            name='Risk Distribution'
        ), row=2, col=2)
        
        fig.update_layout(
            title=f"Area-wise Risk Dashboard - {location_info['state']}",
            height=800,
            showlegend=False
        )
        
        self.logger.complete_step(9, f"Created dashboard for {len(areas)} areas")
        return fig
    
    def create_interactive_map(self, analysis_results: Dict[str, Any]) -> folium.Map:
        """Create interactive Folium map"""
        self.logger.start_step(10)
        
        location_info = analysis_results['location_info']
        center = location_info['center']
        
        # Create base map
        m = folium.Map(
            location=[center[1], center[0]],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add areas as markers
        for area in location_info.get('areas', []):
            # Determine marker color based on risk factors
            risk_factors = area.get('risk_factors', [])
            if any(factor in risk_factors for factor in ['very_low_lying', 'poor_drainage', 'flood_prone']):
                color = 'red'
                icon = 'exclamation-triangle'
            elif any(factor in risk_factors for factor in ['low_lying', 'coastal', 'river_proximity']):
                color = 'orange'
                icon = 'warning'
            else:
                color = 'green'
                icon = 'info'
            
            popup_text = f"""
            <b>{area['name']}</b><br>
            Type: {area['type']}<br>
            Elevation: {area['elevation']}m<br>
            Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}
            """
            
            folium.Marker(
                location=[area['coords'][1], area['coords'][0]],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=area['name'],
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Add water bodies
        for water_body in location_info.get('water_bodies', []):
            if water_body['type'] == 'river':
                coords = water_body['coords']
                if len(coords) >= 2:
                    folium.PolyLine(
                        locations=[[coord[1], coord[0]] for coord in coords],
                        color='blue',
                        weight=3,
                        opacity=0.8,
                        popup=f"{water_body['name']} ({water_body['type']})"
                    ).add_to(m)
            elif water_body['type'] in ['lake', 'pond']:
                center_coord = water_body['coords'][0]
                radius = water_body.get('width', 100)
                folium.Circle(
                    location=[center_coord[1], center_coord[0]],
                    radius=radius,
                    color='blue',
                    fill=True,
                    fillColor='lightblue',
                    fillOpacity=0.6,
                    popup=f"{water_body['name']} ({water_body['type']})"
                ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Risk Level Legend</b><br>
        <i class="fa fa-exclamation-triangle" style="color:red"></i> Very High Risk<br>
        <i class="fa fa-warning" style="color:orange"></i> Moderate Risk<br>
        <i class="fa fa-info" style="color:green"></i> Low Risk<br>
        <i class="fa fa-tint" style="color:blue"></i> Water Bodies
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        self.logger.complete_step(10, "Created interactive map")
        return m

class StreamlitApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.nlp_processor = NLPQueryProcessor()
        self.logger = WorkflowLogger()
        self.visualizer = AdvancedDataVisualizer(self.logger)
        self.setup_workflow()
    
    def setup_workflow(self):
        """Setup analysis workflow steps"""
        self.logger.add_step("Generate Elevation", "Creating realistic elevation data")
        self.logger.add_step("Calculate Slope", "Computing slope gradients")
        self.logger.add_step("Water Proximity", "Analyzing water body distances")
        self.logger.add_step("Drainage Analysis", "Evaluating drainage capacity")
        self.logger.add_step("Land Use Risk", "Assessing land use patterns")
        self.logger.add_step("Risk Integration", "Combining all risk factors")
        self.logger.add_step("Elevation Mapping", "Creating elevation visualizations")
        self.logger.add_step("Risk Heatmap", "Generating flood risk heatmap")
        self.logger.add_step("Factor Analysis", "Multi-factor risk breakdown")
        self.logger.add_step("Area Dashboard", "Area-wise risk assessment")
        self.logger.add_step("Interactive Map", "Creating interactive map")
    
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="Advanced Flood Risk Analysis System",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .risk-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        .workflow-step {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">Flood Risk Analysis System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("🎯 Analysis Configuration")
            
            # Query input
            query = st.text_area(
                "Enter your analysis query:",
                placeholder="e.g., 'Analyze flood risk for Chennai areas near water bodies'",
                height=100
            )
            
            # Process query
            if query:
                intent = self.nlp_processor.extract_intent(query)
                st.subheader("📊 Query Analysis")
                st.json(intent)
                
                context = self.nlp_processor.generate_response_context(intent)
                st.info(context)
            
            # Manual location selection
            st.subheader("🗺️ Location Selection")
            selected_location = st.selectbox(
                "Choose location:",
                ["Chennai", "Mumbai", "Bangalore", "Delhi"]
            )
            
            # Analysis options
            st.subheader("⚙️ Analysis Options")
            show_elevation = st.checkbox("Show Elevation Map", value=True)
            show_risk_heatmap = st.checkbox("Show Risk Heatmap", value=True)
            show_factor_analysis = st.checkbox("Show Factor Analysis", value=True)
            show_area_dashboard = st.checkbox("Show Area Dashboard", value=True)
            show_interactive_map = st.checkbox("Show Interactive Map", value=True)
            
            # Run analysis button
            run_analysis = st.button("🚀 Run Analysis", type="primary")
        
        # Main content area
        if run_analysis:
            location_to_analyze = intent.get('city', selected_location) if 'intent' in locals() else selected_location
            
            with st.spinner(f"Analyzing flood risk for {location_to_analyze}..."):
                # Run analysis
                analysis_results = self.visualizer.risk_analyzer.calculate_comprehensive_flood_risk(
                    location_to_analyze.lower()
                )
                
                # Progress indicator
                progress_col1, progress_col2 = st.columns([3, 1])
                with progress_col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with progress_col2:
                    progress_percent = st.empty()
                
                # Update progress
                for i in range(len(self.logger.steps)):
                    progress = (i + 1) / len(self.logger.steps)
                    progress_bar.progress(progress)
                    status_text.text(f"Step {i+1}: {self.logger.steps[i]['name']}")
                    progress_percent.text(f"{progress*100:.0f}%")
                
                st.success(f"✅ Analysis completed for {location_to_analyze}!")
                
                # Display results
                location_info = analysis_results['location_info']
                
                # Summary statistics
                st.subheader("📈 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Location", f"{location_info['state']}, {location_info['country']}")
                
                with col2:
                    avg_risk = np.mean(analysis_results['flood_risk'])
                    st.metric("Average Risk", f"{avg_risk:.2f}", f"{avg_risk*100:.1f}%")
                
                with col3:
                    high_risk_areas = np.sum(analysis_results['flood_risk'] > 0.7)
                    total_areas = analysis_results['flood_risk'].size
                    st.metric("High Risk Areas", f"{high_risk_areas:,}", 
                             f"{high_risk_areas/total_areas*100:.1f}%")
                
                with col4:
                    coastal_status = "Coastal" if location_info.get('coastal', False) else "Inland"
                    st.metric("Geography", coastal_status)
                
                # Visualizations
                if show_elevation:
                    st.subheader("🏔️ Elevation Analysis")
                    elevation_fig = self.visualizer.create_elevation_map(analysis_results)
                    st.plotly_chart(elevation_fig, use_container_width=True)
                
                if show_risk_heatmap:
                    st.subheader("🌡️ Flood Risk Heatmap")
                    risk_fig = self.visualizer.create_flood_risk_heatmap(analysis_results)
                    st.plotly_chart(risk_fig, use_container_width=True)
                
                if show_factor_analysis:
                    st.subheader("📊 Multi-Factor Risk Analysis")
                    factor_fig = self.visualizer.create_risk_factor_analysis(analysis_results)
                    st.plotly_chart(factor_fig, use_container_width=True)
                
                if show_area_dashboard:
                    st.subheader("🏘️ Area-wise Risk Dashboard")
                    dashboard_fig = self.visualizer.create_area_risk_dashboard(analysis_results)
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                
                if show_interactive_map:
                    st.subheader("🗺️ Interactive Risk Map")
                    interactive_map = self.visualizer.create_interactive_map(analysis_results)
                    folium_static(interactive_map, width=1400, height=600)
                
                # Workflow log
                with st.expander("📋 Analysis Workflow Log"):
                    st.text("Analysis Steps:")
                    for log_entry in self.logger.logs:
                        st.text(log_entry)
                    
                    st.text("Detailed Workflow:")
                    st.code(self.logger.get_workflow_json(), language='json')
                
                # Download results
                st.subheader("💾 Download Results")
                
                # Prepare download data
                download_data = {
                    'location': location_to_analyze,
                    'analysis_summary': {
                        'average_risk': float(np.mean(analysis_results['flood_risk'])),
                        'max_risk': float(np.max(analysis_results['flood_risk'])),
                        'min_risk': float(np.min(analysis_results['flood_risk'])),
                        'high_risk_percentage': float(np.sum(analysis_results['flood_risk'] > 0.7) / analysis_results['flood_risk'].size * 100)
                    },
                    'areas': location_info.get('areas', []),
                    'water_bodies': location_info.get('water_bodies', []),
                    'workflow_log': self.logger.logs
                }
                
                st.download_button(
                    label="📥 Download Analysis Report (JSON)",
                    data=json.dumps(download_data, indent=2),
                    file_name=f"flood_risk_analysis_{location_to_analyze.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()