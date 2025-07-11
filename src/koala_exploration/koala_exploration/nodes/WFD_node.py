#!/usr/bin/env python3

"""
WFD.py — Wavefront Frontier Detector
=======================================================================
Implementacja algorytmu WFD (Wavefront Frontier Detector) bazującego na:
"Efficient frontier detection for robot exploration" (2014)
Autorzy: Keidar, Matan; Kaminka, Gal A.

WFD używa podejścia opartego na przeszukiwaniu grafu do wykrywania frontierów,
przetwarzając tylko punkty przeskanowane przez sensory robota za pomocą BFS.
"""

import time
import hashlib
import threading
import queue
import multiprocessing as mp
import psutil
import numpy as np
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import ParameterDescriptor
from koala_intefaces.msg import Frontiers, Point2DKoala
from pandas import DataFrame
from sklearn.cluster import KMeans
from koala_exploration.utils.clustering import _clustering_frontiers_Kmeans

class WFDDataBase:
    """
    Klasa bazowa dla danych WFD.
    
    Używana do przechowywania i przetwarzania danych wykrytych przez WFD.
    """
    
    def __init__(self):
        self.frontiers = []
        self.clusters = []
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.resolution = 0.05  # Domyślna rozdzielczość siatki
        self.frame_id = 'map'  # Domyślne ID ramki, może być zmienione w update
        self.cpu_time = 0.0
        self.cpu_usage = 0.0   # Add CPU usage tracking
        self.memory_usage = 0.0  # Add memory usage tracking
        self._lock = threading.Lock()
    
    def update(self, frontiers: List[Tuple[int, int]], clusters: List[List[Tuple[int, int]]], 
               origin_x: float, origin_y: float, resolution: float, cpu_time: float,
               cpu_usage: float = 0.0, memory_usage: float = 0.0) -> None:
        """
        Aktualizuje dane WFD.
        
        Args:
            frontiers: Lista punktów frontierowych
            clusters: Lista klastrów punktów frontierowych
            origin_x: Pozycja x początku siatki
            origin_y: Pozycja y początku siatki
            resolution: Rozdzielczość siatki
            cpu_time: Czas wykonania algorytmu
            cpu_usage: Zużycie CPU podczas wykonania
            memory_usage: Zużycie pamięci podczas wykonania
        """
        with self._lock:
            self.frontiers = frontiers
            self.clusters = clusters
            self.origin_x = origin_x
            self.origin_y = origin_y
            self.resolution = resolution
            self.cpu_time = cpu_time
            self.cpu_usage = cpu_usage
            self.memory_usage = memory_usage
    
    def get_data(self) -> dict:
        """
        Zwraca dane WFD jako słownik.
        
        Returns:
            dict: Słownik z danymi WFD
        """
        with self._lock:
            return {
                'frontiers': self.frontiers.copy(),
                'clusters': [cluster.copy() for cluster in self.clusters],
                'origin_x': self.origin_x,
                'origin_y': self.origin_y,
                'resolution': self.resolution,
                'frame_id': self.frame_id,
                'cpu_time': self.cpu_time,
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage
            }

def detect_frontiers_worker(grid_data, robot_pose, origin_x, origin_y, resolution, result_queue):
    """
    Worker process dla wykrywania frontierów
    """
    try:
        # Monitor resource usage
        process = psutil.Process()
        initial_cpu_time = process.cpu_times().user + process.cpu_times().system
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        grid = np.frombuffer(grid_data['data'], dtype=np.int8).reshape(grid_data['shape'])
        
        # Implementacja algorytmu WFD
        frontiers = _detect_frontiers_wfd_static(grid, robot_pose, origin_x, origin_y, resolution)
        
        # Calculate resource usage
        final_cpu_time = process.cpu_times().user + process.cpu_times().system
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = final_cpu_time - initial_cpu_time
        memory_usage = final_memory - initial_memory
        
        result_queue.put({
            'type': 'frontiers',
            'data': frontiers,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'resolution': resolution,
            'timestamp': time.time(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        })
    except Exception as e:
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'timestamp': time.time()
        })

def clustering_worker(self,frontiers, result_queue):
    """
    Worker process dla klastrowania frontierów
    """
    try:
        clusters = _clustering_frontiers_Kmeans(parent=self,frontiers=frontiers, debug_logging=False, divisor=100.0, min_frontiers=5, max_frontiers=15)
        result_queue.put({
            'type': 'clusters',
            'data': clusters,
            'timestamp': time.time()
        })
    except Exception as e:
        result_queue.put({
            'type': 'error',
            'error': str(e),
            'timestamp': time.time()
        })

def _detect_frontiers_wfd_static(grid: np.ndarray, robot_pose: Optional[Tuple[float, float, float]], 
                                origin_x: float, origin_y: float, resolution: float) -> List[Tuple[int,int]]:
    """
    Statyczna wersja algorytmu WFD do użycia w worker process
    """
    # Sprawdź czy są nieznane obszary
    unknown_count = np.sum(grid == -1)
    if unknown_count == 0:
        return []
    
    rows, cols = grid.shape
    
    # Określ punkt startowy
    start_r, start_c = _get_start_position_static(grid, robot_pose, origin_x, origin_y, resolution)
    
    if start_r is None or start_c is None:
        return []
    
    # Inicjalizuj tablice oznaczania dla algorytmu WFD
    marks = np.zeros((rows, cols), dtype=np.uint8)
    
    # Algorytm WFD 3.1: Główne wykrywanie frontierów
    queue_m = [(start_r, start_c)]
    marks[start_r, start_c] = 1
    
    all_frontiers = []
    
    while queue_m:
        p_r, p_c = queue_m.pop(0)
        
        if marks[p_r, p_c] == 2:
            continue
        
        if _is_frontier_point_static(grid, p_r, p_c):
            new_frontier = _extract_frontier_2d_static(grid, p_r, p_c, marks)
            if new_frontier:
                all_frontiers.extend(new_frontier)
                for fr, fc in new_frontier:
                    marks[fr, fc] = 2
        
        neighbors = _get_8_neighbors_static(p_r, p_c, rows, cols)
        for v_r, v_c in neighbors:
            if marks[v_r, v_c] != 0:
                continue
            
            if _has_open_space_neighbor_static(grid, v_r, v_c, rows, cols):
                queue_m.append((v_r, v_c))
                marks[v_r, v_c] = 1
        
        marks[p_r, p_c] = 2
    
    return all_frontiers

def _get_start_position_static(grid: np.ndarray, robot_pose: Optional[Tuple[float, float, float]], 
                              origin_x: float, origin_y: float, resolution: float) -> Tuple[Optional[int], Optional[int]]:
    """Statyczna wersja _get_start_position"""
    rows, cols = grid.shape
    
    if robot_pose:
        start_c = int((robot_pose[0] - origin_x) / resolution)
        start_r = int((robot_pose[1] - origin_y) / resolution)
        
        if (0 <= start_r < rows and 0 <= start_c < cols and grid[start_r, start_c] == 0):
            return start_r, start_c
        
        known_cells = np.where(grid == 0)
        if len(known_cells[0]) > 0:
            distances = ((known_cells[0] - start_r)**2 + (known_cells[1] - start_c)**2)
            nearest_idx = np.argmin(distances)
            return known_cells[0][nearest_idx], known_cells[1][nearest_idx]
    
    known_cells = np.where(grid == 0)
    if len(known_cells[0]) == 0:
        return None, None
    
    center_idx = len(known_cells[0]) // 2
    return known_cells[0][center_idx], known_cells[1][center_idx]

def _extract_frontier_2d_static(grid: np.ndarray, p_r: int, p_c: int, marks: np.ndarray) -> List[Tuple[int,int]]:
    """Statyczna wersja _extract_frontier_2d"""
    rows, cols = grid.shape
    queue_f = [(p_r, p_c)]
    marks[p_r, p_c] = 3
    
    new_frontier = []
    
    while queue_f:
        q_r, q_c = queue_f.pop(0)
        
        if marks[q_r, q_c] in [2, 4]:
            continue
        
        if _is_frontier_point_static(grid, q_r, q_c):
            new_frontier.append((q_r, q_c))
            
            neighbors = _get_8_neighbors_static(q_r, q_c, rows, cols)
            for w_r, w_c in neighbors:
                if marks[w_r, w_c] in [3, 4, 2]:
                    continue
                
                queue_f.append((w_r, w_c))
                marks[w_r, w_c] = 3
        
        marks[q_r, q_c] = 4
    
    return new_frontier

def _is_frontier_point_static(grid: np.ndarray, r: int, c: int) -> bool:
    """Statyczna wersja _is_frontier_point"""
    if grid[r, c] != 0:
        return False
    
    rows, cols = grid.shape
    neighbors = _get_8_neighbors_static(r, c, rows, cols)
    
    for nr, nc in neighbors:
        if grid[nr, nc] == -1:
            return True
    
    return False

def _has_open_space_neighbor_static(grid: np.ndarray, r: int, c: int, rows: int, cols: int) -> bool:
    """Statyczna wersja _has_open_space_neighbor"""
    neighbors = _get_8_neighbors_static(r, c, rows, cols)
    
    for nr, nc in neighbors:
        if grid[nr, nc] == 0:
            return True
    
    return False

def _get_8_neighbors_static(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Statyczna wersja _get_8_neighbors"""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    
    return neighbors

class WFDExplorerNode(Node):
    """
    Implementacja algorytmu Wavefront Frontier Detector (WFD) z multiprocessing
    """
    
    def __init__(self):
        super().__init__('wfd_explorer_node')
        
        # Parametry konfiguracyjne
        self.declare_parameter('debug_logging', True,
                               ParameterDescriptor(description="Czy włączyć szczegółowe logowanie"))
        self.debug_logging = self.get_parameter('debug_logging').get_parameter_value().bool_value
        
        # Parametry mapy - domyślne wartości
        self.map_shape = (1000, 1000)
        self.map_resolution = 0.05
        self.map_origin = (0.0, 0.0)
        
        # Inicjalizacja bazy danych frontierów
        self.DataBase = WFDDataBase()

        # Dane robota i sensory
        self.robot_position = DataFrame(columns=['x', 'y','yaw' , 'timestamp'])
        self._last_odom_msg = self.get_current_time()
        self.map = None
        
        # Kontrola częstotliwości wykrywania
        self.previous_map_size = None
        self.previous_map_hash = None
        self.last_detection_time = 0.0
        self.detection_interval = 5.0  # 4 sekundy między wykrywaniami
        
        # Multiprocessing setup
        self.manager = mp.Manager()
        self.result_queue = self.manager.Queue()
        self.frontier_process = None
        self.clustering_process = None
        self.pending_frontiers = []
        self.pending_clusters = []
        
        # Threading dla przetwarzania wyników
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.result_thread.start()
        
        # Subskrypcje ROS2
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.robot_pose_callback, 10)
        
        # Publisher
        self.frontier_pub = self.create_publisher(Frontiers, 'WFD/frontiers', 10)
        self.publish_timer = self.create_timer(1.0, self.publish_frontiers)
        
        if self.debug_logging:
            self.get_logger().info("🔍 WFD Explorer z zunifikowaną strukturą zainicjalizowany")
            
    def get_current_time(self) -> float:
        """Zwraca aktualny czas w sekundach od epoki."""
        return self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
      
    def robot_pose_callback(self, msg) -> None:
        """Callback dla topicu /odometry/filtered."""
        current_time = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        if current_time - self._last_odom_msg > 0.05:
            self.robot_position.x = msg.pose.pose.position.x
            self.robot_position.y = msg.pose.pose.position.y
            self.robot_position.yaw = 1.0
            self.robot_position.timestamp = current_time
            self._last_odom_log_time = current_time
        else:
            self._last_odom_msg = current_time
            
    def map_callback(self, msg: OccupancyGrid) -> None:
        """Callback dla topicu mapy zajętości"""
        if self.debug_logging:
            self.get_logger().info("�️  Otrzymano nową mapę...")
            
        self.map = msg
        new_shape = (msg.info.height, msg.info.width)
        new_resolution = msg.info.resolution
        new_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        
        # Sprawdź czy mapa się zmieniła znacząco
        map_bytes = bytes(self.map.data)
        current_hash = hashlib.md5(map_bytes).hexdigest()
        current_time = self.get_current_time()
        
        map_changed = (len(self.map.data) != self.previous_map_size or 
                      current_hash != self.previous_map_hash)
        time_elapsed = current_time - self.last_detection_time > self.detection_interval
        
        # Aktualizuj parametry mapy jeśli się zmieniły
        if new_shape != self.map_shape or new_resolution != self.map_resolution or new_origin != self.map_origin:
            self.map_shape = new_shape
            self.map_resolution = new_resolution
            self.map_origin = new_origin
            
            if self.debug_logging:
                self.get_logger().info(f"📏 Zmieniono rozmiar mapy na {self.map_shape}")
        
        # Sprawdź czy procesy się zakończyły
        processes_finished = True
        if self.frontier_process and self.frontier_process.is_alive():
            processes_finished = False
        if self.clustering_process and self.clustering_process.is_alive():
            processes_finished = False
            
        # Wykrywaj frontiere tylko jeśli mapa się zmieniła i minął wystarczający czas
        if map_changed and time_elapsed and processes_finished:
            self.previous_map_size = len(self.map.data)
            self.previous_map_hash = current_hash
            self.last_detection_time = current_time
            
            if self.debug_logging:
                self.get_logger().info("🔍 WFD: Rozpoczynam wykrywanie frontierów...")
            self.cpu_time_start = self.get_current_time()
            self._start_frontier_detection(msg)
    
    def _start_frontier_detection(self, occupancy_grid: OccupancyGrid):
        """Rozpoczyna asynchroniczne wykrywanie frontierów"""
        # Przygotuj dane dla procesu
        grid, origin_x, origin_y, resolution = self._map_to_numpy(occupancy_grid)
        
        # Konwertuj pozycję robota
        robot_pose = None
        if not self.robot_position.empty:
            last_row = self.robot_position.iloc[-1]
            robot_pose = (float(last_row['x']), float(last_row['y']), float(last_row['yaw']))
        
        # Przygotuj dane do przesłania do procesu
        grid_data = {
            'data': grid.tobytes(),
            'shape': grid.shape
        }
        
        # Uruchom proces wykrywania frontierów
        self.frontier_process = mp.Process(
            target=detect_frontiers_worker,
            args=(grid_data, robot_pose, origin_x, origin_y, resolution, self.result_queue)
        )
        self.frontier_process.start()
    
    def _start_clustering(self, frontiers):
        """Rozpoczyna asynchroniczne klastrowanie"""
        if not frontiers:
            return
            
        self.clustering_process = mp.Process(
            target=clustering_worker,
            args=(self,frontiers, self.result_queue)
        )
        self.clustering_process.start()
    
    def _process_results(self):
        """Wątek przetwarzający wyniki z procesów"""
        while True:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                if result['type'] == 'frontiers':
                    self.pending_frontiers = result['data']
                    # Get CPU usage if available
                    cpu_usage = result.get('cpu_usage', 0.0)
                    memory_usage = result.get('memory_usage', 0.0)
                    
                    if self.debug_logging:
                        self.get_logger().info(
                            f"🔍 Otrzymano {len(self.pending_frontiers)} frontierów, "
                            f"CPU: {cpu_usage:.2f}s, Memory: {memory_usage:.2f}MB"
                        )
                    
                    # Rozpocznij klastrowanie za każdym razem gdy znajdą się frontiere
                    self._start_clustering(self.pending_frontiers)
                    
                elif result['type'] == 'clusters':
                    self.pending_clusters = result['data']
                    self.cpu_time_end = self.get_current_time()
                    
                    # Measure current process resource usage
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_mb = process.memory_info().rss / 1024 / 1024  # MB
                    
                    if self.debug_logging:
                        self.get_logger().info(
                            f"🔍 Otrzymano {len(self.pending_clusters)} klastrów, "
                            f"CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB"
                        )
                    
                    # Aktualizuj bazę danych gdy mamy zarówno frontiere jak i klastry
                    if self.map is not None:
                        _, origin_x, origin_y, resolution = self._map_to_numpy(self.map)
                        self.DataBase.update(
                            frontiers=self.pending_frontiers,
                            clusters=self.pending_clusters,
                            origin_x=origin_x,
                            origin_y=origin_y,
                            resolution=resolution,
                            cpu_time=self.cpu_time_end - self.cpu_time_start,
                            cpu_usage=cpu_percent,
                            memory_usage=memory_mb
                        )
                        
                elif result['type'] == 'error':
                    if self.debug_logging:
                        self.get_logger().error(f"❌ Błąd w procesie: {result['error']}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug_logging:
                    self.get_logger().error(f"❌ Błąd przetwarzania wyników: {e}")
    
    def publish_frontiers(self):
        """Publikuje wykryte frontiere i klastry"""
        data = self.DataBase.get_data()
        points = data['frontiers']
        clusters = data['clusters']
        
        # Tworzenie wiadomości ROS
        msg = Frontiers()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = data['frame_id']
        
        # Oblicz centroidy klastrów i posortuj według X
        cluster_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                # Oblicz centroid klastra
                centroid_y = int(np.mean([p[0] for p in cluster]))  # row -> y
                centroid_x = int(np.mean([p[1] for p in cluster]))  # col -> x
                centroid = [centroid_x, centroid_y]  # [x, y]
                # Znajdź frontier najbliższy centroidowi
                min_dist = float('inf')
                for p in cluster:
                    dist = (p[0] - centroid_y) ** 2 + (p[1] - centroid_x) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        centroid[0] = p[1] # x
                        centroid[1] = p[0] # y

                # Dodaj centroid (możesz też dodać nearest_frontier jeśli chcesz)
                cluster_centroids.append(Point2DKoala(x=int(centroid[0]), y=int(centroid[1])))
        
        # Sortuj klastry według współrzędnej X
        cluster_centroids.sort(key=lambda point: point.x)
        
        # Wypełnij wiadomość
        msg.clusters = cluster_centroids
        msg.frontiers = [Point2DKoala(x=int(p[1]), y=int(p[0])) for p in points]
        
        # Dodaj metadane
        msg.origin_x = float(data['origin_x'])
        msg.origin_y = float(data['origin_y'])
        msg.resolution = float(data['resolution'])
        msg.cpu_time = float(data['cpu_time'])
        msg.cpu_usage = float(data['cpu_usage'])
        msg.memory_usage = float(data['memory_usage'])
        
        # Publikuj
        self.frontier_pub.publish(msg)
        
        if self.debug_logging:
            self.get_logger().info(
                f"🔍 Opublikowano {len(points)} frontierów w {len(cluster_centroids)} klastrach "
                f"(CPU: {data['cpu_time']:.3f}s)"
            )
    
    def _map_to_numpy(self, occupancy_grid: OccupancyGrid) -> Tuple[np.ndarray, float, float, float]:
        """Konwertuje OccupancyGrid do numpy array"""
        data = np.array(occupancy_grid.data, dtype=np.int8)
        grid = data.reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        
        resolution = occupancy_grid.info.resolution
        origin_x = occupancy_grid.info.origin.position.x
        origin_y = occupancy_grid.info.origin.position.y
        
        return grid, origin_x, origin_y, resolution
    
    def destroy_node(self):
        """Czyści zasoby przed zniszczeniem węzła"""
        # Zakończ procesy
        if self.frontier_process and self.frontier_process.is_alive():
            self.frontier_process.terminate()
            self.frontier_process.join(timeout=1.0)
        
        if self.clustering_process and self.clustering_process.is_alive():
            self.clustering_process.terminate()
            self.clustering_process.join(timeout=1.0)
        
        super().destroy_node()

def main(args=None):
    """Główna funkcja uruchamiająca węzeł WFD Explorer."""
    rclpy.init(args=args)
    
    wfd_explorer = WFDExplorerNode()
    
    try:
        rclpy.spin(wfd_explorer)
    except KeyboardInterrupt:
        wfd_explorer.get_logger().info('Zatrzymano przez użytkownika')
    finally:
        wfd_explorer.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()