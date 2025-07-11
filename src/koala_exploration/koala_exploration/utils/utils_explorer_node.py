#!/usr/bin/env python3
"""
utils_explorer.py - Utility functions for FrontierExplorer
=======================================================================
Contains helper functions for visualization, parameter management,
and exploration statistics.
"""

import os
import json
import math
import time
from datetime import datetime
from typing import List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid

from ament_index_python.packages import get_package_share_directory


class ExplorerUtils:
    """Utility class for FrontierExplorer functions"""
    
    def __init__(self, node: Node):
        self.node = node
        self._progress_monitor = {
            'navigation_start_time': None,
            'initial_frontier_count': 0,
            'last_frontier_count': 0,
            'last_check_time': None,
            'total_reduction': 0
        }
        node.get_logger().info("ExplorerUtils initialized")
        return
        
        
    def timer_callback(self) -> None:
        """Updates parameters from ROS2 parameter server"""
        try:
            self.node.CLUSTER_RADIUS = self.node.get_parameter('CLUSTER_RADIUS').get_parameter_value().double_value
            self.node.INFO_RADIUS = self.node.get_parameter('INFO_RADIUS').get_parameter_value().double_value
            self.node.INFO_WEIGHT = self.node.get_parameter('INFO_WEIGHT').get_parameter_value().double_value
            self.node.DIST_WEIGHT = self.node.get_parameter('DIST_WEIGHT').get_parameter_value().double_value
            self.node.BLACKLIST_RADIUS = self.node.get_parameter('BLACKLIST_RADIUS').get_parameter_value().double_value
            self.node.SCORE_THRESHOLD = self.node.get_parameter('SCORE_THRESHOLD').get_parameter_value().double_value
            self.node.TIMER_PERIOD = self.node.get_parameter('TIMER_PERIOD').get_parameter_value().double_value
            self.node.VISUALIZATION_REFRESH_PERIOD = self.node.get_parameter('VISUALIZATION_REFRESH_PERIOD').get_parameter_value().double_value
            self.node.ENABLE_VISUALIZATION = self.node.get_parameter('ENABLE_VISUALIZATION').get_parameter_value().bool_value
            self.node.MARKER_LIFETIME = self.node.get_parameter('MARKER_LIFETIME').get_parameter_value().double_value
            self.node.DEBUG_LOGGING = self.node.get_parameter('DEBUG_LOGGING').get_parameter_value().bool_value
            self.node.MIN_SCORE_IMPROVEMENT = self.node.get_parameter('MIN_SCORE_IMPROVEMENT').get_parameter_value().double_value
            self.node.MAP_SAVE_DIR = self.node.get_parameter('MAP_SAVE_DIR').get_parameter_value().string_value
            self.node.MAP_SAVE_ENABLED = self.node.get_parameter('MAP_SAVE_ENABLED').get_parameter_value().bool_value
            self.node.MAP_FAILED_SAVE_THRESHOLD = self.node.get_parameter('MAP_FAILED_SAVE_THRESHOLD').get_parameter_value().integer_value
            self.node.ADAPTIVE_INFO_GAIN = self.node.get_parameter('ADAPTIVE_INFO_GAIN').get_parameter_value().double_value
            self.node.USE_FFD_SOURCE = self.node.get_parameter('USE_FFD_SOURCE').get_parameter_value().bool_value
            self.node.USE_WFD_SOURCE = self.node.get_parameter('USE_WFD_SOURCE').get_parameter_value().bool_value
            self.node.ENABLE_BENCHMARK = self.node.get_parameter('ENABLE_BENCHMARK').get_parameter_value().bool_value
            
        except Exception as e:
            self.node.get_logger().error(f"❌ Błąd podczas aktualizacji parametrów: {e}")
            return

    def visualize_frontiers(self, frontiers: List[Tuple[int,int]], clusters: List[List[Tuple[int,int]]], 
                           ox: float, oy: float, res: float, frontier_blacklist: set,
                           marker_pub, cell_to_world_func) -> None:
        """Wizualizuje frontiere i klastry w RViz"""
        if not self.node.ENABLE_VISUALIZATION or not marker_pub: # type: ignore
            return
            
        marker_array = MarkerArray()
        marker_array.markers = []
        
        # Usuń poprzednie markery
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Wszystkie frontiere (małe niebieskie punkty)
        if frontiers:
            frontier_marker = Marker()
            frontier_marker.points = []
            frontier_marker.header.frame_id = "map"
            frontier_marker.header.stamp = self.node.get_clock().now().to_msg()
            frontier_marker.ns = "frontiers"
            frontier_marker.id = 0
            frontier_marker.type = Marker.POINTS
            frontier_marker.action = Marker.ADD
            frontier_marker.scale.x = 0.05
            frontier_marker.scale.y = 0.05
            frontier_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
            frontier_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
            
            for cell in frontiers:
                x, y = cell_to_world_func(cell, ox, oy, res)
                point = Point(x=x, y=y, z=0.0)
                frontier_marker.points.append(point)
            
            marker_array.markers.append(frontier_marker)
        
        # Centra klastrów (zielone sfery)
        for i, cluster in enumerate(clusters):
            cluster_marker = Marker()
            cluster_marker.header.frame_id = "map"
            cluster_marker.header.stamp = self.node.get_clock().now().to_msg()
            cluster_marker.ns = "clusters"
            cluster_marker.id = i
            cluster_marker.type = Marker.SPHERE
            cluster_marker.action = Marker.ADD
            cluster_marker.scale.x = 0.3
            cluster_marker.scale.y = 0.3
            cluster_marker.scale.z = 0.3
            cluster_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
            
            x, y = cell_to_world_func(cluster[0], ox, oy, res)
            cluster_marker.pose.position.x = x
            cluster_marker.pose.position.y = y
            cluster_marker.pose.position.z = 0.15
            cluster_marker.pose.orientation.w = 1.0
            
            # Kolor zależny od statusu blacklisty
            if cluster[0] in frontier_blacklist:
                cluster_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
                cluster_marker.ns = "blacklisted_clusters"
            else:
                cluster_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            
            marker_array.markers.append(cluster_marker)
        
        # Publikuj markery
        marker_pub.publish(marker_array)

    def visualize_blacklist(self, frontier_blacklist: set, ox: float, oy: float, res: float,
                           blacklist_pub, cell_to_world_func) -> None:
        """Wizualizuje obszary na blackliście"""
        if not self.node.ENABLE_VISUALIZATION or not blacklist_pub: # type: ignore
            return
            
        marker_array = MarkerArray()
        marker_array.markers = []
        
        # Usuń poprzednie markery
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        if not frontier_blacklist:
            blacklist_pub.publish(marker_array)
            return
        
        # Blacklistowane komórki jako czerwone kwadraty
        blacklist_marker = Marker()
        blacklist_marker.points = []
        blacklist_marker.header.frame_id = "map"
        blacklist_marker.header.stamp = self.node.get_clock().now().to_msg()
        blacklist_marker.ns = "blacklist_cells"
        blacklist_marker.id = 0
        blacklist_marker.type = Marker.CUBE_LIST
        blacklist_marker.action = Marker.ADD
        blacklist_marker.scale.x = res
        blacklist_marker.scale.y = res
        blacklist_marker.scale.z = 0.1
        blacklist_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
        blacklist_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
        
        # Dodaj wszystkie blacklistowane komórki
        for cell in frontier_blacklist:
            x, y = cell_to_world_func(cell, ox, oy, res)
            point = Point(x=x, y=y, z=0.05)
            blacklist_marker.points.append(point)
        
        marker_array.markers.append(blacklist_marker)
        blacklist_pub.publish(marker_array)

    def visualize_frontier_info(self, frontier_scores: List[dict], 
                               best_cell: Optional[Tuple[int,int]], 
                               info_pub, eval_timer) -> None:
        
        """Wizualizuje informacje o frontierach z cost penalty"""
        if not self.node.ENABLE_VISUALIZATION or not info_pub: # type: ignore
            return
            
        marker_array = MarkerArray()
        
        # Usuń poprzednie markery
        delete_marker = Marker()
        marker_array.markers = []
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Dodaj informację o trybie continuous evaluation
        status_marker = Marker()
        status_marker.header.frame_id = "map"
        status_marker.header.stamp = self.node.get_clock().now().to_msg()
        status_marker.ns = "continuous_eval_status"
        status_marker.id = 999
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.scale.z = 0.5
        
        # Status continuous evaluation
        is_continuous_active = eval_timer is not None
        cluster_count = len(frontier_scores)
        
        if is_continuous_active:
            status_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Zielony
            status_text = f"CONTINUOUS: ON\nClusters: {cluster_count}"
        else:
            status_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # Żółty
            status_text = f"CONTINUOUS: OFF\nClusters: {cluster_count}"
        
        status_marker.text = status_text
        status_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
        status_marker.pose.position.x = 0.0
        status_marker.pose.position.y = 0.0
        status_marker.pose.position.z = 2.0
        status_marker.pose.orientation.w = 1.0
        
        marker_array.markers.append(status_marker)
        
        # Reszta markerów frontierów
        for i, info in enumerate(frontier_scores):
            cell = info['cell']
            x, y = info['position']
            
            # Tekst z informacjami o frontierze
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.node.get_clock().now().to_msg()
            text_marker.ns = "frontier_info"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.2
            
            # Kolor: czerwony dla najlepszego, żółty dla pozostałych
            if cell == best_cell:
                text_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                text_marker.scale.z = 0.3
            else:
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
            
            text_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.5
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = (
                f"Score: {info['total_score']:.1f}\n"
                f"Info: {info['info_gain']}\n"
                f"Dist: {info['path_length']:.1f}m\n"
                f"Dist_score: {info['score_dist']:.1f}\n"
            )
            
            marker_array.markers.append(text_marker)
            
            # Dodaj strzałkę dla najlepszego frontiera
            if cell == best_cell:
                arrow_marker = Marker()
                arrow_marker.header.frame_id = "map"
                arrow_marker.header.stamp = self.node.get_clock().now().to_msg()
                arrow_marker.ns = "best_frontier"
                arrow_marker.id = 0
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.scale.x = 0.5
                arrow_marker.scale.y = 0.1
                arrow_marker.scale.z = 0.1
                arrow_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                arrow_marker.lifetime.sec = int(self.node.MARKER_LIFETIME) # type: ignore
                arrow_marker.pose.position.x = x
                arrow_marker.pose.position.y = y
                arrow_marker.pose.position.z = 0.3
                arrow_marker.pose.orientation.w = 1.0
                
                marker_array.markers.append(arrow_marker)
        
        info_pub.publish(marker_array)

    def save_exploration_summary(self, exploration_stats: dict, map_to_numpy_func) -> None:
        """Zapisuje podsumowanie eksploracji do pliku JSON"""
        try:
            if not os.path.exists(self.node.MAP_SAVE_DIR): # type: ignore
                os.makedirs(self.node.MAP_SAVE_DIR) # type: ignore
            
            # Zaktualizuj czas końcowy
            exploration_stats['end_time'] = time.time()
            
            # Oblicz całkowity czas eksploracji
            if exploration_stats['start_time']:
                total_time = exploration_stats['end_time'] - exploration_stats['start_time']
                exploration_stats['total_time_seconds'] = total_time
                exploration_stats['total_time_formatted'] = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
            
            # Dodaj dodatkowe statystyki
            exploration_stats['success_rate'] = (
                exploration_stats['successful_navigations'] / 
                max(exploration_stats['navigation_attempts'], 1) * 100
            )
            
            # Oblicz średnią prędkość
            if exploration_stats['total_time_seconds'] > 0:
                exploration_stats['average_speed_ms'] = (
                    exploration_stats['total_distance'] / 
                    exploration_stats['total_time_seconds']
                )
            exploration_stats['total_distance'] = round(exploration_stats['total_distance'], 2)
            
            # Parametry eksploracji
            exploration_stats['exploration_parameters'] = { 
                'CLUSTER_RADIUS': self.node.CLUSTER_RADIUS, # type: ignore
                'INFO_RADIUS': self.node.INFO_RADIUS, # type: ignore
                'INFO_WEIGHT': self.node.INFO_WEIGHT, # type: ignore
                'DIST_WEIGHT': self.node.DIST_WEIGHT, # type: ignore
                'BLACKLIST_RADIUS': self.node.BLACKLIST_RADIUS, # type: ignore
                'SCORE_THRESHOLD': self.node.SCORE_THRESHOLD, # type: ignore
                'TIMER_PERIOD': self.node.TIMER_PERIOD, # type: ignore
                'MIN_SCORE_IMPROVEMENT': self.node.MIN_SCORE_IMPROVEMENT, # type: ignore
                'MAP_FAILED_SAVE_THRESHOLD': self.node.MAP_FAILED_SAVE_THRESHOLD, # type: ignore
                'ADAPTIVE_INFO_GAIN': self.node.ADAPTIVE_INFO_GAIN, # type: ignore
            }
            
            # Parametry nawigacji (używaj już zebranych z exploration_stats)
            # exploration_stats['navigation_parameters'] już zawiera parametry zebrane podczas startu

            # Przygotuj dane do zapisu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exploration_summary_{timestamp}.json"
            filepath = os.path.join(self.node.MAP_SAVE_DIR, filename) # type: ignore
            
            # Zapisz do pliku JSON
            with open(filepath, 'w') as f:
                json.dump(exploration_stats, f, indent=2)
            
            self.node.get_logger().info(f"📊 Podsumowanie eksploracji zapisane do: {filepath}")
            
            # Wyświetl podsumowanie w logach
            self.log_exploration_summary(exploration_stats)
            
        except Exception as e:
            self.node.get_logger().error(f"❌ Błąd podczas zapisywania podsumowania: {e}")

    def log_exploration_summary(self, stats: dict) -> None:
        """Wyświetla podsumowanie eksploracji w logach"""
        self.node.get_logger().info("📊 ===== PODSUMOWANIE EKSPLORACJI =====")
        self.node.get_logger().info(f"⏱️  Całkowity czas: {stats.get('total_time_formatted', 'N/A')}")
        self.node.get_logger().info(f"📏 Pokonal odległość: {stats['total_distance']:.2f} m")
        self.node.get_logger().info(f"🎯 Odwiedzone frontiere: {stats['frontiers_visited']}")
        self.node.get_logger().info(f"🚀 Próby nawigacji: {stats['navigation_attempts']}")
        self.node.get_logger().info(f"✅ Udane nawigacje: {stats['successful_navigations']}")
        self.node.get_logger().info(f"❌ Nieudane nawigacje: {stats['failed_navigations']}")
        self.node.get_logger().info(f"📈 Wskaźnik sukcesu: {stats.get('success_rate', 0):.1f}%")
        self.node.get_logger().info(f"🏃 Średnia prędkość: {stats.get('average_speed_ms', 0):.2f} m/s")
        self.node.get_logger().info(f"🗺️  Punkty trasy: {len(stats['path_points'])}")
        self.node.get_logger().info("📊 ====================================")

    def extract_parameter_value(self, param) -> Any:
        """Wyciąga wartość z Parameter message z obsługą wszystkich typów"""
        if param.type == ParameterType.PARAMETER_STRING:
            return param.string_value
        elif param.type == ParameterType.PARAMETER_DOUBLE:
            return param.double_value
        elif param.type == ParameterType.PARAMETER_INTEGER:
            return param.integer_value
        elif param.type == ParameterType.PARAMETER_BOOL:
            return param.bool_value
        elif param.type == ParameterType.PARAMETER_BYTE_ARRAY:
            return list(param.byte_array_value)
        elif param.type == ParameterType.PARAMETER_BOOL_ARRAY:
            return list(param.bool_array_value)
        elif param.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            return list(param.integer_array_value)
        elif param.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return list(param.double_array_value)
        elif param.type == ParameterType.PARAMETER_STRING_ARRAY:
            return list(param.string_array_value)
        elif param.type == ParameterType.PARAMETER_NOT_SET:
            return None
        else:
            self.node.get_logger().warning(f"Nieznany typ parametru: {param.type}")
            return None

    def get_nav2_parameters(self) -> dict:
        """Odczytuje parametry z węzłów Nav2"""
        nav_params = {}
        
        # Lista węzłów Nav2 i ich parametrów
        nav2_nodes = {}
         # Wczytaj listę węzłów i parametrów z pliku JSON
        json_path = os.path.join(get_package_share_directory('koala_exploration'), "params", "nav2_nodes.json")
        with open(json_path, "r") as f:
            nav2_nodes = json.load(f)
            
        for node_name, param_names in nav2_nodes.items():
            service_name = f'/{node_name}/get_parameters'
            client = self.node.create_client(GetParameters, service_name)
            
            # ✅ ZWIĘKSZONY timeout - niektóre węzły mogą potrzebować więcej czasu
            if client.wait_for_service(timeout_sec=2.0):
                request = GetParameters.Request()
                request.names = param_names
                
                try:
                    future = client.call_async(request)
                    rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
                    
                    result = future.result()
                    if result and hasattr(result, 'values'):
                        response = result
                        for i, param in enumerate(response.values):
                            if i < len(param_names):
                                param_value = self.extract_parameter_value(param)
                                if param_value is not None:
                                    nav_params[f"{node_name}.{param_names[i]}"] = param_value
                                else:
                                    # ✅ DODAJ wartość NULL zamiast pomijania
                                    nav_params[f"{node_name}.{param_names[i]}"] = "NOT_SET"
                                    
                        self.node.get_logger().info(f"✅ Odczytano {len(param_names)} parametrów z {node_name}")
                    else:
                        self.node.get_logger().warning(f"⚠️  Brak odpowiedzi z {node_name}")
                        
                except Exception as e:
                    self.node.get_logger().warning(f"Błąd odczytywania z {node_name}: {e}")
            else:
                self.node.get_logger().warning(f"Service {service_name} niedostępny")
                # ✅ DODAJ placeholder dla niedostępnych węzłów
                for param_name in param_names:
                    nav_params[f"{node_name}.{param_name}"] = "SERVICE_UNAVAILABLE"

        # ✅ DODAJ podsumowanie
        total_params = len(nav_params)
        available_nodes = len([k for k in nav_params.keys() if not nav_params[k] in ["SERVICE_UNAVAILABLE", "NOT_SET"]])

        self.node.get_logger().info(f"📊 Nav2 Parameters: {total_params} parametrów, {available_nodes} dostępnych")

        return nav_params

    def save_path_visualization(self, exploration_stats: dict, occupancy_map: OccupancyGrid, 
                            map_to_numpy_func, frontiers: List[Tuple[int,int]] | None = None) -> None:
        """Zapisuje wizualizację trasy do pliku PNG"""
        try:
            if not exploration_stats['path_points'] or not occupancy_map:
                return
                
            # Pobierz dane mapy
            grid, ox, oy, res = map_to_numpy_func(occupancy_map)
            
            # Przygotuj figurę
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Wyświetl mapę jako tło
            # Konwertuj mapę do formatu wyświetlania
            display_grid = np.zeros_like(grid, dtype=float)
            display_grid[grid == -1] = 0.5  # nieznane -> szare
            display_grid[grid == 0] = 1.0   # wolne -> białe  
            display_grid[grid > 0] = 0.0    # przeszkody -> czarne
            
            # Wyświetl mapę (odwróć oś Y dla prawidłowego wyświetlania)
            ax.imshow(display_grid, cmap='gray', origin='lower', 
                     extent=[ox, ox + grid.shape[1]*res, 
                            oy, oy + grid.shape[0]*res])
            
            # Narysuj frontiery na kolor cyan
            if frontiers:
                frontier_x = []
                frontier_y = []
                for (r, c) in frontiers:
                    x = ox + c * res
                    y = oy + r * res
                    frontier_x.append(x)
                    frontier_y.append(y)
                
                ax.scatter(frontier_x, frontier_y, c='cyan', s=10, alpha=0.8, label='Frontiery')
            
            # Narysuj trasę
            if len(exploration_stats['path_points']) > 1:
                x_coords = [p['x'] for p in exploration_stats['path_points']]
                y_coords = [p['y'] for p in exploration_stats['path_points']]
                
                # Trasa jako linia
                ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8, label='Trasa robota')
                
                # Punkt startowy
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
                
                # Punkt końcowy
                ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Koniec')
            
            # Dodaj informacje
            ax.set_title(f"Mapa eksploracji - Dystans: {exploration_stats['total_distance']:.1f}m")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # Zapisz do pliku
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exploration_path_{timestamp}.png"
            filepath = os.path.join(self.node.MAP_SAVE_DIR, filename) # type: ignore
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.node.get_logger().info(f"🗺️  Wizualizacja trasy zapisana do: {filepath}")
            
        except Exception as e:
            self.node.get_logger().error(f"❌ Błąd podczas zapisywania wizualizacji: {e}")            
