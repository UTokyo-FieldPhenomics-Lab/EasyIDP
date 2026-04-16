"""
Subplot Generation Core Module.

Reference logic from QGIS plugin `fieldShape.py` but ported to geopandas/shapely.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
from loguru import logger


class SubplotGenerator:
    """
    Generator for creating subplots within a field boundary.

    Uses Minimum Area Rectangle (MAR) to determine field orientation and
    divides it into a grid based on rows/cols or dimensions.
    """

    def __init__(self):
        pass

    def load_boundary(self, shp_path: str) -> Optional[gpd.GeoDataFrame]:
        """
        Load boundary shapefile and validate it.
        """
        try:
            gdf = gpd.read_file(shp_path)
            if len(gdf) != 1:
                logger.error("Boundary file must contain exactly one polygon.")
                return None
            
            geom_type = gdf.geometry.iloc[0].geom_type
            if geom_type not in ['Polygon', 'MultiPolygon']:
                logger.error(f"Invalid geometry type: {geom_type}")
                return None
                
            return gdf
        except Exception as e:
            logger.error(f"Failed to load boundary: {e}")
            return None

    def generate(
        self,
        boundary_gdf: gpd.GeoDataFrame,
        rows: int,
        cols: int,
        x_space: float,
        y_space: float,
        output_path: Optional[str] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Generate subplots.

        Parameters
        ----------
        boundary_gdf : gpd.GeoDataFrame
            DataFrame containing the single boundary polygon.
        rows : int
            Number of rows (vertical divisions).
        cols : int
            Number of columns (horizontal divisions).
        x_space : float
            Spacing between columns (meters).
        y_space : float
            Spacing between rows (meters).
        output_path : str, optional
            Path to save the generated shapefile.

        Returns
        -------
        gpd.GeoDataFrame
            Appended DataFrame with generated subplots.
        """
        if len(boundary_gdf) != 1:
            raise ValueError("Boundary GDF must contain exactly one feature")

        polygon = boundary_gdf.geometry.iloc[0]
        crs = boundary_gdf.crs

        # 1. Calculate Minimum Area Rectangle (MAR)
        # shapely minimum_rotated_rectangle returns a Polygon
        mar = polygon.minimum_rotated_rectangle
        
        # Get coordinates of MAR
        # coords is list of (x, y). Last one repeats first.
        coords = list(mar.exterior.coords)
        if len(coords) < 4:
            logger.error("Invalid MAR geometry")
            return None
        
        # Determine "bottom" and "left" vectors similar to QGIS plugin
        # The logic in QGIS plugin assumes rows along long edge, cols along short edge.
        # But here we should probably stick to geometric "width" and "height".
        
        # Let's find the longest edge of MAR to define "width" direction (X-axis equivalent)
        # and short edge as "height" direction (Y-axis equivalent).
        
        p0 = np.array(coords[0])
        p1 = np.array(coords[1])
        p2 = np.array(coords[2])
        
        edge1_vec = p1 - p0
        edge2_vec = p2 - p1
        
        len1 = np.linalg.norm(edge1_vec)
        len2 = np.linalg.norm(edge2_vec)
        
        if len1 >= len2:
            # edge1 is "width" (bottom), edge2 is "height" (left)
            width_vec = edge1_vec
            height_vec = edge2_vec
            width_len = len1
            height_len = len2
            start_p = p0
        else:
            # edge2 is "width", edge1 is is "height" (right side actually, but vector direction matters)
            # To simplify, let's just pick p1 as start, edge2 as width, and vector from p1->p2
            # And height vector would be p2->p3 or similar orthogonal.
            # Actually, MAR is a rectangle. 
            # Vector p1->p2 and p2->p3 are orthogonal.
            width_vec = edge2_vec
            # edge after p2 is p2->p3. Since p1-p0 is orthogonal to p2-p1, then p0-p1 is parallel to p3-p2.
            # Let's use p1 as origin.
            
            # Re-eval corners to find canonical origin and axes
            # For consistency with QGIS logic: "rows along long edge (x), cols along short edge (y)"
            
            # Let's just use the logic:
            # Longest side is the primary axis (Column distribution/Width).
            # Shortest side is secondary axis (Row distribution/Height).
            
            # Wait, QGIS code says: "rows always along long edge(x)".
            # But normally rows stack vertically (along Y). So "rows" count means dividing Y axis.
            # QGIS code: "rowsLbl": "Vertical divisions (rows)" -> Divides Y axis. Correct.
            # So "rows" count determines divisions along the Short Edge (Height).
            # "cols" count determines divisions along the Long Edge (Width).
            
            width_vec = edge2_vec
            height_vec = p0 - p1 # Vector pointing "up" from p1
             # Check if p0-p1 is indeed orthogonal. Yes it is.
            width_len = len2
            height_len = len1
            start_p = p1

        # Normalized vectors
        width_dir = width_vec / width_len
        height_dir = height_vec / height_len

        # Calculate cell dimensions
        # total_width = width_len
        # total_height = height_len
        
        # width is divided by cols
        # height is divided by rows
        
        # Formula: cell_w = (TotalW - (cols-1)*space_x) / cols
        cell_w = (width_len - (cols - 1) * x_space) / cols
        cell_h = (height_len - (rows - 1) * y_space) / rows

        if cell_w <= 0 or cell_h <= 0:
            logger.error("Spacing too large, resulting valid cell size is negative.")
            return None

        # Generate subplots
        polys = []
        data = []
        
        # Step vectors
        step_x = width_dir * (cell_w + x_space)
        step_y = height_dir * (cell_h + y_space)
        
        # Vector for cell size
        vec_cw = width_dir * cell_w
        vec_ch = height_dir * cell_h
        
        for r in range(rows):
            for c in range(cols):
                # Origin of current cell
                origin = start_p + (c * step_x) + (r * step_y)
                
                # 4 corners
                # p_bl = origin (Bottom-Left in local coords)
                # p_br = origin + vec_cw
                # p_tr = origin + vec_cw + vec_ch
                # p_tl = origin + vec_ch
                
                p_items = [
                    origin,
                    origin + vec_cw,
                    origin + vec_cw + vec_ch,
                    origin + vec_ch,
                    origin # close
                ]
                
                poly = Polygon(p_items)
                polys.append(poly)
                
                # attributes: id, row, col (1-based)
                data.append({
                    'id': len(polys),
                    'row': r + 1,
                    'col': c + 1
                })
        
        # Create GeoDataFrame
        gdf_out = gpd.GeoDataFrame(data, geometry=polys, crs=crs)
        
        # Save if path provided
        if output_path:
            gdf_out.to_file(output_path, encoding='utf-8')
            logger.info(f"Saved subplots to {output_path}")
            
        return gdf_out
        
    def calculate_optimal_rotation(self, boundary_gdf: gpd.GeoDataFrame) -> Optional[float]:
        """
        Calculate rotation angle to make the longest edge of MAR horizontal.
        
        Returns
        -------
        float
            Rotation angle in degrees.
        """
        if len(boundary_gdf) != 1:
            return None
            
        polygon = boundary_gdf.geometry.iloc[0]
        mar = polygon.minimum_rotated_rectangle
        coords = list(mar.exterior.coords)
        if len(coords) < 4:
            return None
            
        # Determine long edge
        p0 = np.array(coords[0])
        p1 = np.array(coords[1])
        p2 = np.array(coords[2])
        
        edge1 = p1 - p0
        edge2 = p2 - p1
        
        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)
        
        if len1 >= len2:
            # edge1 is long edge
            vec = edge1
        else:
            # edge2 is long edge
            vec = edge2
            
        # Calculate angle of this vector
        angle_rad = math.atan2(vec[1], vec[0])
        angle_deg = math.degrees(angle_rad)
        
        # We probably want to align long edge to X axis (0 degrees)
        # So we rotate by -angle
        # But MapCanvas.set_rotation usually takes positive = clockwise or counter-clockwise?
        # QGIS setRotation(angle) rotates the MAP VIEW.
        # If map is rotated by +angle, the content rotates by -angle.
        # If vector is at 30 deg, we want to rotate VIEW by 30 deg to make it horizontal?
        # Yes.
        
        return angle_deg
