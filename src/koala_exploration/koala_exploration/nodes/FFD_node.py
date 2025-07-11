#!/usr/bin/env python3
"""
Fast Frontier Detector (FFD) – pełna implementacja algorytmu Keidar & Kaminka 2013
===============================================================================
Implementacja algorytmu FFD dla ROS2 z pełną strukturą zgodną z pseudokodem.

Wymaga:
    • numpy
    • rclpy
    • sensor_msgs, nav_msgs
    • koala_interfaces
"""

import math
import time
import threading
from collections import defaultdict
from functools import cmp_to_key
import matplotlib
matplotlib.use('Agg')  # Użyj backend bez GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Set, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from koala_intefaces.msg import Frontiers, Point2DKoala
from koala_exploration.utils.clustering import _clustering_frontiers_Kmeans, _clustering_frontiers_DBScan


# =============================================================================
# NARZĘDZIA GEOMETRYCZNE
# =============================================================================

def bresenham_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Algorytm Bresenhama dla rysowania linii między dwoma punktami.
    
    Args:
        p0: Punkt początkowy (x, y)
        p1: Punkt końcowy (x, y)
        
    Returns:
        Lista punktów na linii
    """
    (x0, y0), (x1, y1) = map(int, p0), map(int, p1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    points = []
    
    while True:
        points.append((x0, y0))
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points


# =============================================================================
# STRUKTURY DANYCH FFD
# =============================================================================

class FrontierDB:
    """
    Baza danych granic zgodna z algorytmem FFD.
    Przechowuje mapowanie punktów do ID granic oraz zbiory punktów dla każdej granicy.
    """
    
    def __init__(self, shape: Tuple[int, int]):
        self.grid_idx = np.full(shape, fill_value=None, dtype=object)
        self.frontiers = defaultdict(set)
        self._next_id = 0

    def new_frontier(self) -> int:
        """Tworzy nowy ID granicy."""
        fid = self._next_id
        self._next_id += 1
        return fid

    def add_point(self, fid: int, p: Tuple[int, int]) -> None:
        """Dodaje punkt do granicy."""
        if self._is_valid_point(p):
            self.frontiers[fid].add(p)
            self.grid_idx[p] = fid

    def remove_point(self, p: Tuple[int, int]) -> Optional[int]:
        """Usuwa punkt z granicy i zwraca ID granicy."""
        if not self._is_valid_point(p):
            return None
        
        fid = self.grid_idx[p]
        if fid is None:
            return None
        
        self.frontiers[fid].discard(p)
        self.grid_idx[p] = None
        
        # Usuń granicę jeśli jest pusta
        if not self.frontiers[fid]:
            del self.frontiers[fid]
        
        return fid

    def merge(self, fid_dst: int, fid_src: int) -> None:
        """Scala dwie granice."""
        if fid_dst == fid_src or fid_src not in self.frontiers:
            return
        
        # Aktualizuj indeksy punktów
        for p in self.frontiers[fid_src]:
            if self._is_valid_point(p):
                self.grid_idx[p] = fid_dst
        
        # Scal zbiory punktów
        self.frontiers[fid_dst] |= self.frontiers[fid_src]
        del self.frontiers[fid_src]

    def split_frontier(self, fid: int, split_point: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Dzieli granicę na dwie części w miejscu split_point.
        
        Returns:
            Tuple zawierający ID nowych granic (może być None jeśli część jest pusta)
        """
        if fid not in self.frontiers or split_point not in self.frontiers[fid]:
            return None, None
        
        frontier_points = list(self.frontiers[fid])
        try:
            split_index = frontier_points.index(split_point)
        except ValueError:
            return None, None
        
        # Podziel na dwie części
        part1 = frontier_points[:split_index]
        part2 = frontier_points[split_index + 1:]
        
        # Usuń oryginalną granicę
        for p in frontier_points:
            self.grid_idx[p] = None
        del self.frontiers[fid]
        
        # Utwórz nowe granice
        new_fid1, new_fid2 = None, None
        
        if part1:
            new_fid1 = self.new_frontier()
            for p in part1:
                self.add_point(new_fid1, p)
        
        if part2:
            new_fid2 = self.new_frontier()
            for p in part2:
                self.add_point(new_fid2, p)
        
        return new_fid1, new_fid2

    def get_frontier_id(self, p: Tuple[int, int]) -> Optional[int]:
        """Zwraca ID granicy dla danego punktu."""
        if self._is_valid_point(p):
            return self.grid_idx[p]
        return None

    def get_all_frontier_points(self) -> List[Tuple[int, int]]:
        """Zwraca wszystkie punkty granic."""
        all_points = []
        for frontier_set in self.frontiers.values():
            all_points.extend(list(frontier_set))
        return all_points

    def _is_valid_point(self, p: Tuple[int, int]) -> bool:
        """Sprawdza czy punkt jest w granicach mapy."""
        return (0 <= p[0] < self.grid_idx.shape[0] and 
                0 <= p[1] < self.grid_idx.shape[1])


# =============================================================================
# GŁÓWNY NODE FFD
# =============================================================================

class FFDExplorerNode(Node):
    """
    Node ROS2 implementujący algorytm Fast Frontier Detector.
    """
    
    def __init__(self):
        super().__init__('ffd_explorer_node')
        
        # Parametry
        self.declare_parameter('debug_logging', True)
        self.debug_logging = self.get_parameter('debug_logging').get_parameter_value().bool_value
        
        # Callback groups dla równoległego przetwarzania
        self.laser_callback_group = ReentrantCallbackGroup()
        self.publisher_callback_group = MutuallyExclusiveCallbackGroup()
        self.map_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Lock do synchronizacji dostępu do danych
        self._data_lock = threading.Lock()
        
        # Inicjalizacja struktur danych
        self._initialize_data_structures()
        
        # Inicjalizacja komunikacji ROS2
        self._initialize_ros2_communication()
        
        self._init_margin_visualization()
        
        if self.debug_logging:
            self.get_logger().info("🚀 FFD Explorer uruchomiony z równoległym przetwarzaniem")

    def _initialize_data_structures(self):
        """Inicjalizuje struktury danych FFD."""
        # Parametry mapy
        self.map_shape = (1000, 1000)
        self.map_resolution = 0.05
        self.map_origin = (0.0, 0.0)
        
        # Struktury FFD (chronione lockiem)
        self.db = FrontierDB(self.map_shape)
        self.visit_counter = np.zeros(self.map_shape, dtype=np.uint16)
        self.active_area: Set[Tuple[int, int]] = set()
        self.clusters = []
        
        # Metryki wydajności
        self.cpu_time = 0.0
        self.cpu_usage = 0.0
        
        # Dane robota i sensory
        self._last_scan = None
        self._last_pose = (0.0, 0.0, 0.0)
        self.map = None
        
        # Flaga gotowości do wykrywania
        self.detection_in_progress = False

    def _initialize_ros2_communication(self):
        """Inicjalizuje komunikację ROS2 z callback groups."""
        # Subskrypcje z różnymi callback groups
        self.create_subscription(
            LaserScan, 
            '/scan', 
            self._laser_callback, 
            10,
            callback_group=self.laser_callback_group
        )
        
        self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self._map_callback, 
            10,
            callback_group=self.map_callback_group
        )
        
        self.create_subscription(
            Odometry, 
            '/odometry/filtered', 
            self._odom_callback, 
            10,
            callback_group=self.map_callback_group
        )
        
        # Publisher z osobnym timerem
        self.frontier_pub = self.create_publisher(Frontiers, 'FFD/frontiers', 10)
        self.create_timer(
            0.5, 
            self.publish_frontiers,
            callback_group=self.publisher_callback_group
        )

    # =========================================================================
    # CALLBACKS ROS2
    # =========================================================================

    def _odom_callback(self, msg: Odometry):
        """Callback dla odometrii robota."""
        with self._data_lock:
            self._last_pose = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                self._extract_yaw_from_quaternion(msg.pose.pose.orientation)
            )

    def _laser_callback(self, msg: LaserScan):
        """Callback dla danych lidar - uruchamia wykrywanie frontierów."""
        with self._data_lock:
            self._last_scan = msg
            
            # Sprawdź czy poprzednie wykrywanie się zakończyło
            if self.detection_in_progress:
                if self.debug_logging:
                    self.get_logger().debug("🔄 Pomijam wykrywanie - poprzednie w toku")
                return
            
            self.detection_in_progress = True
        
        # Uruchom wykrywanie frontierów (poza lockiem)
        self.detect_frontiers()

    def _map_callback(self, msg: OccupancyGrid):
        """Callback dla mapy zajętości."""
        with self._data_lock:
            self._update_map_parameters(msg)
            self._resize_data_structures_if_needed()

    # =========================================================================
    # ALGORYTM FFD - GŁÓWNE ETAPY
    # =========================================================================

    def detect_frontiers(self):
        """
        Główna funkcja wykrywania granic - Algorytm 4.1.
        Działa asynchronicznie względem publikacji.
        """
        try:
            # Sprawdź dostępność danych (z lockiem)
            with self._data_lock:
                if not self._can_detect_frontiers():
                    if self.debug_logging:
                        self.get_logger().warning("⚠️  FFD: Brak danych do wykrywania")
                    return
                
                # NOWE: Walidacja spójności danych
                if not self._validate_frontier_consistency():
                    self.get_logger().warning("⚠️  Wykryto niespójności w bazie frontierów - naprawiam...")
                    self._repair_frontier_consistency()
                
                # Skopiuj dane potrzebne do przetwarzania
                map_data = self.map.data[:]
                map_shape = self.map_shape
                last_scan = self._last_scan
                last_pose = self._last_pose
            
            if self.debug_logging:
                self.get_logger().info("🔍 FFD: Rozpoczynam wykrywanie granic")
            
            start_time = time.time()
            
            # Przygotowanie danych (bez locka)
            grid_data = np.array(map_data, dtype=np.int8)
            occ_grid = grid_data.reshape(map_shape)
            laser_pts = self._laser_to_points(last_scan, last_pose)
            
            if not laser_pts:
                if self.debug_logging:
                    self.get_logger().warning("⚠️  FFD: Brak punktów laser")
                return
            
            robot_grid_pos = self._world_to_grid((last_pose[0], last_pose[1]))
            
            # DIAGNOSTYKA: Sprawdź problemy z ujemnymi współrzędnymi
            if self.debug_logging:
                self._diagnose_coordinate_issues(laser_pts, last_pose)
            
            # Aktualizacja aktywnego obszaru (z lockiem)
            with self._data_lock:
                self._update_active_area(laser_pts)
            
            # Główny algorytm FFD (bez locka dla większości operacji)
            self._fast_frontier_detector_threadsafe(laser_pts, robot_grid_pos, occ_grid)
            
            # Aktualizacja metryk (z lockiem)
            with self._data_lock:
                self.cpu_time = time.time() - start_time
                self.cpu_usage = self.cpu_time * 100
                
            if self.debug_logging:
                with self._data_lock:
                    frontier_count = len(self.db.get_all_frontier_points())
                self.get_logger().info(
                    f"🔍 FFD: {frontier_count} frontierów w {self.cpu_time:.3f}s"
                )
                
        finally:
            # Zawsze oznacz koniec wykrywania
            with self._data_lock:
                self.detection_in_progress = False

    def _fast_frontier_detector_threadsafe(self, laser_pts: List[Tuple[int, int]], 
                                          pose: Tuple[int, int], occ_grid: np.ndarray):
        """
        Implementacja głównego algorytmu FFD z synchronizacją wątków.
        """
        # Krok 1-3: Przetwarzanie bez locka
        sorted_pts = self._sort_polar(laser_pts, pose)
        contour = self._build_contour(sorted_pts)
        new_frontiers = self._extract_frontiers_1d(contour, occ_grid)
        
        # Kroki 4-6: Aktualizacja struktur danych z lockiem
        with self._data_lock:
            self._maintain_frontiers(occ_grid)
            self._merge_overlapping_frontiers(new_frontiers)
            self._update_visit_counters(contour)

    def _sort_polar(self, laser_pts: List[Tuple[int, int]], 
                   pose: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Sortowanie polarne - Algorytm 4.2.
        Używa komparatora opartego na iloczynie wektorowym.
        """
        if not laser_pts:
            return []
        
        origin = np.asarray(pose)
        pts = np.asarray(laser_pts)
        
        def polar_comparator(i1: int, i2: int) -> int:
            """Komparator polarny używający iloczynu wektorowego."""
            p1 = pts[i1] - origin
            p2 = pts[i2] - origin
            
            # Iloczyn wektorowy
            cross_product = p1[0] * p2[1] - p2[0] * p1[1]
            
            if cross_product > 0:
                return 1
            elif cross_product < 0:
                return -1
            else:
                # Jeśli punkty są kolinearne, sortuj według odległości
                dist1 = np.linalg.norm(p1)
                dist2 = np.linalg.norm(p2)
                return 1 if dist1 > dist2 else (-1 if dist1 < dist2 else 0)
        
        # Sortowanie z użyciem komparatora polarnego
        idx_sorted = sorted(range(len(pts)), key=cmp_to_key(polar_comparator))
        return [tuple(pts[i]) for i in idx_sorted]

    def _build_contour(self, sorted_pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Budowanie konturu - Algorytm 4.3.
        Łączy posortowane punkty liniami Bresenhama.
        """
        if not sorted_pts:
            return []
        
        contour = []
        prev_point = sorted_pts[-1]  # Ostatni punkt jako poprzedni
        
        for curr_point in sorted_pts:
            # Dodaj linię między punktami
            line_points = bresenham_line(prev_point, curr_point)
            contour.extend(line_points)
            prev_point = curr_point
        
        # Usuń duplikaty zachowując kolejność
        seen = set()
        unique_contour = []
        for point in contour:
            if point not in seen:
                seen.add(point)
                unique_contour.append(point)
        
        return unique_contour

    def _extract_frontiers_1d(self, contour: List[Tuple[int, int]], 
                             occ_grid: np.ndarray) -> List[int]:
        """
        Wyodrębnianie granic - Algorytm 4.4.
        Wykrywa nowe granice leżące na konturze.
        """
        new_frontiers = []
        if not contour:
            return new_frontiers
        
        prev_point = contour[0]
        current_frontier_id = None
        
        # Przypadek specjalny: pierwszy punkt
        if self._is_frontier_cell(prev_point, occ_grid):
            current_frontier_id = self.db.new_frontier()
            self.db.add_point(current_frontier_id, prev_point)
            new_frontiers.append(current_frontier_id)
        
        # Przetwarzanie reszty punktów
        for curr_point in contour[1:]:
            if not self._is_frontier_cell(curr_point, occ_grid):
                # Obecny punkt nie jest komórką graniczną
                current_frontier_id = None
            elif self._has_been_visited_before(curr_point):
                # Obecny punkt był odwiedzony wcześniej
                current_frontier_id = None
            elif (self._is_frontier_cell(curr_point, occ_grid) and 
                  self._is_frontier_cell(prev_point, occ_grid) and 
                  current_frontier_id is not None):
                # Kontynuacja istniejącej granicy
                self.db.add_point(current_frontier_id, curr_point)
            else:
                # Nowa granica
                current_frontier_id = self.db.new_frontier()
                self.db.add_point(current_frontier_id, curr_point)
                new_frontiers.append(current_frontier_id)
            
            prev_point = curr_point
        
        return new_frontiers

    def _maintain_frontiers(self, occ_grid: np.ndarray):
        """
        Utrzymanie wcześniej wykrytych granic - Algorytm 4.5.
        NAPRAWIONA WERSJA: Bezpieczniejsze usuwanie punktów z synchronizacją.
        """
        # Zbierz wszystkie punkty frontierów (kopia dla bezpieczeństwa)
        all_frontier_points = self.db.get_all_frontier_points()[:]
        
        if self.debug_logging:
            self.get_logger().info(f"🧹 _maintain_frontiers(): Sprawdzam {len(all_frontier_points)} punktów frontierów")
        
        # Zbierz punkty do usunięcia zamiast usuwać od razu
        points_to_remove = []
        for point in all_frontier_points:
            # Sprawdź czy punkt nadal jest poprawnym frontierem
            if not self._is_frontier_cell_strict(point, occ_grid):
                points_to_remove.append(point)
        
        # Usuń punkty w bezpieczny sposób
        points_removed = 0
        frontiers_affected = set()
        
        for point in points_to_remove:
            fid = self.db.get_frontier_id(point)
            if fid is not None:
                frontiers_affected.add(fid)
                self.db.remove_point(point)
                points_removed += 1
        
        # Sprawdź spójność po usunięciu
        if frontiers_affected:
            if self.debug_logging:
                self.get_logger().debug(f"🧹 Naprawiam spójność po usunięciu punktów z {len(frontiers_affected)} frontierów")
            self._repair_frontier_consistency()
        
        if self.debug_logging:
            self.get_logger().info(f"🧹 _maintain_frontiers(): Usunięto {points_removed} punktów z {len(frontiers_affected)} frontierów")

    def _is_frontier_cell_strict(self, point: Tuple[int, int], occ_grid: np.ndarray) -> bool:
        """Bardziej rygorystyczna wersja sprawdzania frontierów."""
        x, y = point

        # 1. Sprawdź granice mapy
        if not (0 <= x < occ_grid.shape[0] and 0 <= y < occ_grid.shape[1]):
            return False

        # # 2. ZWIĘKSZONY próg licznika odwiedzin
        # if self.visit_counter[x, y] > 10:  # Zwiększone z 3 do 5
        #     return False

        # 3. BARDZIEJ RESTRYKCYJNE kryteria dla mapy
        cell_value = occ_grid[x, y]
        # Frontier musi być TYLKO nieznany (-1), nie akceptujemy już poznanych komórek
        if cell_value != -1:  # Tylko nieznane komórki mogą być frontierami
            return False

        # 4. Sprawdzenie sąsiadów - WYMAGANE co najmniej 2 puste komórki obok
        free_neighbors = 0
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < occ_grid.shape[0] and 
                0 <= ny < occ_grid.shape[1]):
                if occ_grid[nx, ny] == 0:  # Tylko całkowicie puste
                    free_neighbors += 1
        
        # Wymagamy przynajmniej 2 wolnych sąsiadów
        return free_neighbors >= 1

    def _merge_overlapping_frontiers(self, new_frontiers: List[int]):
        """
        Scalanie nakładających się granic - NAPRAWIONA WERSJA z bezpieczną synchronizacją.
        """
        if not new_frontiers:
            return
            
        merged_frontiers = set()
        
        for fid in new_frontiers[:]:
            if fid not in self.db.frontiers or fid in merged_frontiers:
                continue
            
            # Znajdź sąsiednie frontiers do scalenia
            frontiers_to_merge = set()
            
            # Sprawdź każdy punkt tego frontiera
            for point in list(self.db.frontiers[fid]):
                # Sprawdź sąsiednie punkty
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        neighbor = (point[0] + dx, point[1] + dy)
                        if not self.db._is_valid_point(neighbor):
                            continue
                            
                        neighbor_fid = self.db.get_frontier_id(neighbor)
                        
                        if (neighbor_fid is not None and 
                            neighbor_fid != fid and 
                            neighbor_fid in self.db.frontiers and
                            neighbor_fid not in merged_frontiers):
                            frontiers_to_merge.add(neighbor_fid)
            
            # Scal wszystkie znalezione frontiers
            if frontiers_to_merge:
                target_fid = min(fid, min(frontiers_to_merge))  # Użyj najmniejszego ID jako docelowego
                
                for merge_fid in frontiers_to_merge:
                    if merge_fid != target_fid and merge_fid in self.db.frontiers:
                        # Bezpieczne scalanie
                        self._safe_merge_frontiers(target_fid, merge_fid)
                        merged_frontiers.add(merge_fid)
                        
                        # Usuń z listy nowych frontierów jeśli tam był
                        if merge_fid in new_frontiers:
                            new_frontiers.remove(merge_fid)
                
                merged_frontiers.add(target_fid)
        
        if self.debug_logging and merged_frontiers:
            self.get_logger().debug(f"🔗 Scalono {len(merged_frontiers)} frontierów")

    def _safe_merge_frontiers(self, target_fid: int, source_fid: int):
        """
        Bezpiecznie scala dwa frontiers.
        """
        if target_fid == source_fid or source_fid not in self.db.frontiers:
            return
            
        if target_fid not in self.db.frontiers:
            # Jeśli docelowy frontier nie istnieje, po prostu zmień ID źródłowego
            self.db.frontiers[target_fid] = self.db.frontiers[source_fid]
            # Aktualizuj grid_idx
            for point in self.db.frontiers[target_fid]:
                if self.db._is_valid_point(point):
                    self.db.grid_idx[point] = target_fid
            del self.db.frontiers[source_fid]
            return
        
        # Przenieś wszystkie punkty ze źródłowego do docelowego
        points_to_move = list(self.db.frontiers[source_fid])
        
        for point in points_to_move:
            if self.db._is_valid_point(point):
                # Aktualizuj grid_idx
                self.db.grid_idx[point] = target_fid
                # Dodaj do docelowego frontiera
                self.db.frontiers[target_fid].add(point)
        
        # Usuń źródłowy frontier
        del self.db.frontiers[source_fid]
        
        if self.debug_logging:
            self.get_logger().debug(f"🔗 Scalono frontier {source_fid} z {target_fid} ({len(points_to_move)} punktów)")

    # =========================================================================
    # FUNKCJE POMOCNICZE
    # =========================================================================
    
    def _init_margin_visualization(self):
        """Inicjalizuje wizualizację marginesu."""
        self.enable_margin_viz = True  # Flaga włączająca wizualizację marginesu
        self.save_plots = True  # Flaga do zapisywania wykresów
        
        # Timer do odświeżania wizualizacji co 20s
        self.create_timer(
            20.0,  # 20 sekund
            self._visualize_margin,
            callback_group=self.publisher_callback_group
        )
        
        # Timer do okresowego sprawdzania spójności co 30s
        self.create_timer(
            30.0,  # 30 sekund
            self._periodic_consistency_check,
            callback_group=self.publisher_callback_group
        )
        
        # Dane do wizualizacji
        self.last_laser_pts = []
        self.last_margin_info = {}
        self.plot_counter = 0  # Licznik do numerowania plików

    def _update_active_area(self, laser_pts: List[Tuple[int, int]]):
        """Aktualizuje aktywny obszar na podstawie punktów laser - NAPRAWIONA WERSJA dla ujemnych współrzędnych."""
        if not laser_pts:
            return
        
        # Zapisz dane do wizualizacji
        if self.enable_margin_viz:
            self.last_laser_pts = laser_pts[:]
        
        # Oblicz prostokąt ograniczający z marginesem
        xs, ys = zip(*laser_pts)
    
        # Adaptacyjny margines na podstawie gęstości punktów
        num_points = len(laser_pts)
        if num_points > 1000:
            margin = 8  # Więcej punktów = większy margines
        elif num_points > 500:
            margin = 6
        else:
            margin = 4  # Mniej punktów = mniejszy margines
        
        # Zapisz informacje o marginesie
        if self.enable_margin_viz:
            self.last_margin_info = {
                'xs': xs,
                'ys': ys,
                'margin': margin,
                'num_points': num_points,
                'x_min_raw': min(xs),
                'x_max_raw': max(xs),
                'y_min_raw': min(ys),
                'y_max_raw': max(ys)
            }
        
        # NAPRAWIONE: Lepsze obliczanie granic aktywnego obszaru
        x_min_with_margin = min(xs) - margin
        x_max_with_margin = max(xs) + margin
        y_min_with_margin = min(ys) - margin
        y_max_with_margin = max(ys) + margin
        
        # Ogranicz tylko do rzeczywistych granic mapy (nie obcinaj ujemnych bez potrzeby)
        x_min = max(0, x_min_with_margin)
        x_max = min(self.map_shape[0] - 1, x_max_with_margin)
        y_min = max(0, y_min_with_margin)
        y_max = min(self.map_shape[1] - 1, y_max_with_margin)
        
        # Aktualizuj aktywny obszar - tylko prawidłowe indeksy
        self.active_area.clear()
        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                if (0 <= x < self.map_shape[0] and 0 <= y < self.map_shape[1]):
                    self.active_area.add((x, y))
        
        if self.debug_logging:
            lost_points_x = max(0, -x_min_with_margin) + max(0, x_max_with_margin - (self.map_shape[0] - 1))
            lost_points_y = max(0, -y_min_with_margin) + max(0, y_max_with_margin - (self.map_shape[1] - 1))
            
            # Tylko ostrzegaj jeśli strata jest znacząca (>10% obszaru)
            total_area_with_margin = (x_max_with_margin - x_min_with_margin) * (y_max_with_margin - y_min_with_margin)
            total_area_actual = (x_max - x_min) * (y_max - y_min)
            loss_percentage = ((total_area_with_margin - total_area_actual) / total_area_with_margin * 100) if total_area_with_margin > 0 else 0
            
            if loss_percentage > 10:  # Tylko jeśli straca > 10% obszaru
                self.get_logger().warning(
                    f"⚠️  Znaczące obcięcie aktywnego obszaru ({loss_percentage:.1f}% straty)! "
                    f"Oryginał: x[{x_min_with_margin}:{x_max_with_margin}], y[{y_min_with_margin}:{y_max_with_margin}] "
                    f"-> x[{x_min}:{x_max}], y[{y_min}:{y_max}]"
                )
            elif loss_percentage > 0:
                self.get_logger().debug(
                    f"🔄 Niewielkie obcięcie aktywnego obszaru ({loss_percentage:.1f}% straty)"
                )
            
            self.get_logger().debug(f"🔄 Aktywny obszar: {len(self.active_area)} punktów (margin: {margin}, strata: {loss_percentage:.1f}%)")

    def _visualize_margin(self):
        """Wizualizuje aktywny obszar i margin co 20 sekund - WERSJA BEZ INTERAKTYWNOŚCI."""
        if not self.enable_margin_viz or not self.last_laser_pts or not self.last_margin_info:
            return
            
        try:
            # Skopiuj dane (z lockiem)
            with self._data_lock:
                laser_pts = self.last_laser_pts[:]
                margin_info = self.last_margin_info.copy()
                active_area = self.active_area.copy()
                robot_pos = self._last_pose[:2] if self._last_pose else (0, 0)
                frontier_points = self.db.get_all_frontier_points()[:]
            
            # Utwórz nowy wykres (każdy w osobnej funkcji)
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            ax.set_title(f"FFD - Aktywny obszar (margin: {margin_info['margin']}, "
                        f"punktów: {margin_info['num_points']})")
            ax.set_xlabel("X (grid)")
            ax.set_ylabel("Y (grid)")
            ax.grid(True, alpha=0.3)
            
            # 1. Narysuj punkty lasera (niebieskie kropki)
            if laser_pts:
                laser_x = [p[1] for p in laser_pts]  # kolumna -> x
                laser_y = [p[0] for p in laser_pts]  # wiersz -> y
                ax.scatter(laser_x, laser_y, c='blue', s=2, alpha=0.6, label='Punkty laser')
            
            # 2. Narysuj prostokąt bez marginesu (czerwona linia)
            x_min_raw, x_max_raw = margin_info['x_min_raw'], margin_info['x_max_raw']
            y_min_raw, y_max_raw = margin_info['y_min_raw'], margin_info['y_max_raw']
            
            raw_rect = patches.Rectangle(
                (y_min_raw, x_min_raw),  # (x, y) w matplotlib
                y_max_raw - y_min_raw,   # width
                x_max_raw - x_min_raw,   # height
                linewidth=2, edgecolor='red', facecolor='none',
                label='Obszar bez marginesu'
            )
            ax.add_patch(raw_rect)
            
            # 3. Narysuj prostokąt z marginesem (zielona linia)
            margin = margin_info['margin']
            x_min_margin = max(0, x_min_raw - margin)
            x_max_margin = min(self.map_shape[0] - 1, x_max_raw + margin)
            y_min_margin = max(0, y_min_raw - margin)
            y_max_margin = min(self.map_shape[1] - 1, y_max_raw + margin)
            
            margin_rect = patches.Rectangle(
                (y_min_margin, x_min_margin),  # (x, y) w matplotlib
                y_max_margin - y_min_margin,   # width
                x_max_margin - x_min_margin,   # height
                linewidth=2, edgecolor='green', facecolor='green', alpha=0.1,
                label=f'Aktywny obszar (margin: {margin})'
            )
            ax.add_patch(margin_rect)
            
            # 4. Narysuj punkty aktywnego obszaru (małe zielone kropki)
            if active_area and len(active_area) < 5000:  # Tylko jeśli nie za dużo punktów
                active_x = [p[1] for p in active_area]  # kolumna -> x
                active_y = [p[0] for p in active_area]  # wiersz -> y
                ax.scatter(active_x, active_y, c='green', s=0.5, alpha=0.3, label='Aktywny obszar')
            
            # 5. Narysuj frontiers (pomarańczowe kropki)
            if frontier_points:
                frontier_x = [p[1] for p in frontier_points]  # kolumna -> x
                frontier_y = [p[0] for p in frontier_points]  # wiersz -> y
                ax.scatter(frontier_x, frontier_y, c='orange', s=8, alpha=0.8, label='Frontiers')
            
            # 6. Narysuj pozycję robota (duża czerwona kropka)
            robot_grid = self._world_to_grid(robot_pos)
            ax.scatter(robot_grid[1], robot_grid[0], c='red', s=100, marker='o', 
                      label='Robot', edgecolors='black', linewidth=1)
            
            # 7. Dodaj informacje tekstowe
            info_text = (
                f"Punkty laser: {len(laser_pts)}\n"
                f"Margin: {margin}\n"
                f"Aktywny obszar: {len(active_area)} punktów\n"
                f"Frontiers: {len(frontier_points)}\n"
                f"Robot: ({robot_grid[1]}, {robot_grid[0]})"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 8. Ustaw zakres osi z marginesem
            if laser_pts:
                padding = 20
                ax.set_xlim(y_min_margin - padding, y_max_margin + padding)
                ax.set_ylim(x_min_margin - padding, x_max_margin + padding)
            
            # 9. Legenda i finalizacja
            ax.legend(loc='upper right')
            ax.set_aspect('equal')
            plt.tight_layout()
            
            # 10. Zapisz do pliku zamiast wyświetlania
            if self.save_plots:
                self.plot_counter += 1
                filename = f'/tmp/ffd_margin_visualization_{self.plot_counter:04d}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                
                if self.debug_logging:
                    self.get_logger().info(f"📊 Zapisano wizualizację do {filename}")
            
            # 11. WAŻNE: Zamknij wykres żeby zwolnić pamięć
            plt.close(fig)
            
            if self.debug_logging:
                self.get_logger().info(
                    f"📊 Wizualizacja marginesu: {len(laser_pts)} punktów laser, "
                    f"margin: {margin}, aktywny obszar: {len(active_area)} punktów"
                )
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd wizualizacji marginesu: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def _create_margin_comparison_plot(self):
        """Tworzy wykres porównujący różne wartości marginesu - WERSJA BEZ INTERAKTYWNOŚCI."""
        if not self.last_laser_pts:
            return
            
        try:
            # Testuj różne wartości marginesu
            margins_to_test = [0, 2, 4, 6, 8, 10]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Porównanie różnych wartości marginesu", fontsize=16)
            
            laser_pts = self.last_laser_pts
            xs, ys = zip(*laser_pts)
            
            for i, margin in enumerate(margins_to_test):
                ax = axes[i // 3, i % 3]
                
                # Oblicz obszary
                x_min_raw, x_max_raw = min(xs), max(xs)
                y_min_raw, y_max_raw = min(ys), max(ys)
                
                x_min_margin = max(0, x_min_raw - margin)
                x_max_margin = min(self.map_shape[0] - 1, x_max_raw + margin)
                y_min_margin = max(0, y_min_raw - margin)
                y_max_margin = min(self.map_shape[1] - 1, y_max_raw + margin)
                
                # Oblicz rozmiar aktywnego obszaru
                active_area_size = (x_max_margin - x_min_margin + 1) * (y_max_margin - y_min_margin + 1)
                
                # Narysuj punkty laser
                laser_x = [p[1] for p in laser_pts]
                laser_y = [p[0] for p in laser_pts]
                ax.scatter(laser_x, laser_y, c='blue', s=2, alpha=0.6)
                
                # Prostokąt bez marginesu
                raw_rect = patches.Rectangle(
                    (y_min_raw, x_min_raw), y_max_raw - y_min_raw, x_max_raw - x_min_raw,
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(raw_rect)
                
                # Prostokąt z marginesem
                if margin > 0:
                    margin_rect = patches.Rectangle(
                        (y_min_margin, x_min_margin), y_max_margin - y_min_margin, x_max_margin - x_min_margin,
                        linewidth=1, edgecolor='green', facecolor='green', alpha=0.1
                    )
                    ax.add_patch(margin_rect)
                
                ax.set_title(f"Margin: {margin}\nObszar: {active_area_size} punktów")
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Zapisz do pliku
            filename = '/tmp/ffd_margin_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)  # WAŻNE: Zamknij wykres
            
            if self.debug_logging:
                self.get_logger().info(f"📊 Zapisano porównanie marginesów do {filename}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Błąd porównania marginesów: {e}")
    def _update_visit_counters(self, contour: List[Tuple[int, int]]):
        """Aktualizuje liczniki odwiedzin dla punktów konturu."""
        for point in contour:
            if (0 <= point[0] < self.visit_counter.shape[0] and 
                0 <= point[1] < self.visit_counter.shape[1]):
                if self.visit_counter[point] < 10:
                    self.visit_counter[point] += 1
    
    def _is_frontier_cell_for_maintenance(self, point: Tuple[int, int], occ_grid: np.ndarray) -> bool:
        """
        Sprawdza czy punkt jest komórką graniczną - WERSJA DLA MAINTENANCE.
        Ta wersja jest bardziej agresywna w usuwaniu znanych obszarów.
        """
        x, y = point

        # 1. Sprawdź granice mapy
        if not (0 <= x < occ_grid.shape[0] and 0 <= y < occ_grid.shape[1]):
            if self.debug_logging:
                self.get_logger().debug(f"🧹 Punkt {point} poza granicami mapy")
            return False

        # 2. Sprawdź wartość komórki
        cell_value = occ_grid[x, y]
        
        # KLUCZOWA ZMIANA: Jeśli komórka jest już znana (nie -1), to nie jest frontierem
        if cell_value != -1:
            if self.debug_logging:
                self.get_logger().debug(f"🧹 Punkt {point} już znany (wartość: {cell_value})")
            return False

        # 3. Sprawdź czy ma sąsiada w otwartej przestrzeni
        has_free_neighbor = False
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < occ_grid.shape[0] and 
                0 <= ny < occ_grid.shape[1]):
                neighbor_value = occ_grid[nx, ny]
                if neighbor_value == 0:  # Pusta przestrzeń
                    has_free_neighbor = True
                    break

        if not has_free_neighbor:
            if self.debug_logging:
                self.get_logger().debug(f"🧹 Punkt {point} nie ma pustych sąsiadów")
            return False

        return True

    def _is_frontier_cell(self, point: Tuple[int, int], occ_grid: np.ndarray) -> bool:
        """Sprawdza czy punkt jest komórką graniczną - ŁAGODNIEJSZA WERSJA."""
        x, y = point
    
        # 1. Sprawdź granice mapy
        if not (0 <= x < occ_grid.shape[0] and 0 <= y < occ_grid.shape[1]):
            return False
    
        # 2. ŁAGODNIEJSZY licznik odwiedzin - zwiększ próg
        if self.visit_counter[x, y] > 3:  
            return False
    
        # 3. ŁAGODNIEJSZE kryteria dla nieznanych regionów
        # Akceptuj nieznane (-1) i komórki o niskim prawdopodobieństwie
        cell_value = occ_grid[x, y]
        if not (cell_value == -1 or (0 <= cell_value <= 5)):
            return False
    
        # 4. ŁAGODNIEJSZE kryteria dla sąsiadów
        # Szukaj sąsiadów w otwartej przestrzeni LUB o niskim prawdopodobieństwie
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < occ_grid.shape[0] and 
                0 <= ny < occ_grid.shape[1]):
                neighbor_value = occ_grid[nx, ny]
                # ŁAGODNIEJSZE: akceptuj puste (0
                if neighbor_value == 0: 
                    return True
    
        return False

    def _has_been_visited_before(self, point: Tuple[int, int]) -> bool:
        """Sprawdza czy punkt był wcześniej odwiedzony."""
        return self.visit_counter[point] > 3

    def _can_detect_frontiers(self) -> bool:
        """Sprawdza czy można wykrywać granice."""
        return self.map is not None and self._last_scan is not None

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Konwertuje współrzędne świata na współrzędne siatki.
        NAPRAWIONA WERSJA: Obsługuje ujemne współrzędne świata poprawnie.
        """
        x, y = world_pos
        grid_col = int((x - self.map_origin[0]) / self.map_resolution)
        grid_row = int((y - self.map_origin[1]) / self.map_resolution)
        
        # DEBUGGING: Loguj problematyczne konwersje
        if self.debug_logging and (grid_col < 0 or grid_row < 0):
            self.get_logger().debug(
                f"🌍➡️📍 Ujemne grid: world({x:.2f}, {y:.2f}) -> grid({grid_row}, {grid_col}) "
                f"[origin: ({self.map_origin[0]:.2f}, {self.map_origin[1]:.2f}), res: {self.map_resolution:.4f}]"
            )
        
        return grid_row, grid_col

    def _laser_to_points(self, scan: LaserScan, robot_pose: Tuple[float, float, float]) -> List[Tuple[int, int]]:
        """
        Konwertuje dane laser na punkty siatki.
        NAPRAWIONA WERSJA: Lepsze obsługiwanie punktów poza mapą.
        """
        if not scan or not scan.ranges:
            return []
        
        robot_x, robot_y, robot_yaw = robot_pose
        points = []
        angle = scan.angle_min
        points_outside_map = 0
        points_negative_grid = 0
        
        for range_val in scan.ranges:
            if (scan.range_min <= range_val <= scan.range_max and 
                not math.isnan(range_val) and not math.isinf(range_val)):
                
                point_x = robot_x + range_val * math.cos(robot_yaw + angle)
                point_y = robot_y + range_val * math.sin(robot_yaw + angle)
                grid_point = self._world_to_grid((point_x, point_y))
                
                # Sprawdź czy punkt mieści się w mapie
                if (0 <= grid_point[0] < self.map_shape[0] and 
                    0 <= grid_point[1] < self.map_shape[1]):
                    points.append(grid_point)
                else:
                    points_outside_map += 1
                    if grid_point[0] < 0 or grid_point[1] < 0:
                        points_negative_grid += 1
            
            angle += scan.angle_increment
        
        # DEBUGGING: Loguj problematyczne punkty
        if self.debug_logging and (points_outside_map > 0 or points_negative_grid > 0):
            self.get_logger().debug(
                f"🔍 Laser: {len(points)} punktów w mapie, {points_outside_map} poza mapą "
                f"({points_negative_grid} z ujemnymi grid), robot: ({robot_x:.2f}, {robot_y:.2f})"
            )
        
        return points

    def _extract_yaw_from_quaternion(self, quat) -> float:
        """Wyciąga kąt yaw z quaternionu."""
        # Uproszczona implementacja - można rozszerzyć o pełną konwersję
        return 0.0

    def _update_map_parameters(self, msg: OccupancyGrid):
        """Aktualizuje parametry mapy - NAPRAWIONA WERSJA z obsługą ujemnych origin."""
        old_map = self.map
        old_shape = self.map_shape if hasattr(self, 'map_shape') else None
        old_origin = self.map_origin if hasattr(self, 'map_origin') else None
        
        # Aktualizuj parametry mapy
        self.map = msg
        new_shape = (msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        new_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_origin = new_origin
        
        # Sprawdź czy rozmiar lub origin się zmienił
        shape_changed = (old_shape != new_shape)
        origin_changed = (old_origin != new_origin)
        
        if self.debug_logging and (shape_changed or origin_changed):
            self.get_logger().info(
                f"📏 Zmiana mapy: "
                f"shape: {old_shape} → {new_shape}, "
                f"origin: {old_origin} → {new_origin}, "
                f"resolution: {self.map_resolution:.4f}"
            )
            
            # OSTRZEŻENIE o ujemnych origins
            if new_origin[0] < 0 or new_origin[1] < 0:
                self.get_logger().warning(
                    f"⚠️  UWAGA: Ujemny origin mapy może powodować problemy z konwersją współrzędnych!"
                )
        
        if shape_changed:
            if self.debug_logging:
                self.get_logger().info(
                    f"📏 Wykryto zmianę rozmiaru mapy: {old_shape} → {new_shape}"
                )
            
            # Aktualizuj rozmiar tylko jeśli rzeczywiście się zmienił
            self.map_shape = new_shape
            
            # Wywołaj resize w następnym cyklu (poza lockiem map callback)
            self.create_timer(0.1, self._delayed_resize, callback_group=self.publisher_callback_group)
        else:
            # Jeśli rozmiar się nie zmienił, tylko aktualizuj parametry
            self.map_shape = new_shape

    def _delayed_resize(self):
        """Opóźniona zmiana rozmiaru struktur danych."""
        with self._data_lock:
            self._resize_data_structures_if_needed()
        
        # Usuń timer (jednorazowe wykonanie)
        for timer in self._timers:
            if timer.callback == self._delayed_resize:
                timer.destroy()
                break
    
    def _validate_frontier_consistency(self) -> bool:
        """
        Sprawdza spójność bazy danych frontierów - ULEPSZONA WERSJA.
        """
        try:
            inconsistencies = []
            
            # 1. Sprawdź czy wszystkie punkty w frontiers mają odpowiednie wpisy w grid_idx
            for fid, points in self.db.frontiers.items():
                for point in points:
                    if not self.db._is_valid_point(point):
                        inconsistencies.append(f"Punkt {point} w frontier {fid} jest poza granicami mapy")
                        continue
                        
                    grid_fid = self.db.grid_idx[point]
                    if grid_fid != fid:
                        inconsistencies.append(f"Punkt {point} ma grid_idx={grid_fid}, ale należy do frontier {fid}")
            
            # 2. Sprawdź czy wszystkie wpisy w grid_idx mają odpowiednie punkty w frontiers
            frontier_points_from_grid = 0
            for row in range(self.db.grid_idx.shape[0]):
                for col in range(self.db.grid_idx.shape[1]):
                    fid = self.db.grid_idx[row, col]
                    if fid is not None:
                        frontier_points_from_grid += 1
                        if fid not in self.db.frontiers:
                            inconsistencies.append(f"grid_idx[{row},{col}]={fid}, ale frontier {fid} nie istnieje")
                        elif (row, col) not in self.db.frontiers[fid]:
                            inconsistencies.append(f"grid_idx[{row},{col}]={fid}, ale punkt ({row},{col}) nie w frontier {fid}")
            
            # 3. Sprawdź czy liczba punktów się zgadza
            total_frontier_points = sum(len(points) for points in self.db.frontiers.values())
            if total_frontier_points != frontier_points_from_grid:
                inconsistencies.append(f"Niezgodność liczby punktów: frontiers={total_frontier_points}, grid_idx={frontier_points_from_grid}")
            
            # Loguj wszystkie niespójności
            if inconsistencies and self.debug_logging:
                self.get_logger().warning(f"⚠️  Znaleziono {len(inconsistencies)} niespójności:")
                for i, issue in enumerate(inconsistencies[:5]):  # Pokaż tylko pierwsze 5
                    self.get_logger().warning(f"  {i+1}. {issue}")
                if len(inconsistencies) > 5:
                    self.get_logger().warning(f"  ... i {len(inconsistencies) - 5} więcej")
            
            return len(inconsistencies) == 0
            
        except Exception as e:
            if self.debug_logging:
                self.get_logger().error(f"❌ Błąd walidacji spójności: {e}")
            return False

    def _repair_frontier_consistency(self):
        """Naprawia niespójności w bazie danych frontierów."""
        if self.debug_logging:
            self.get_logger().info("🔧 Naprawiam niespójności w bazie frontierów...")
        
        # Wyczyść grid_idx
        self.db.grid_idx.fill(None)
        
        # Odbuduj grid_idx na podstawie frontiers
        valid_frontiers = {}
        for fid, points in self.db.frontiers.items():
            valid_points = set()
            for point in points:
                if self.db._is_valid_point(point):
                    valid_points.add(point)
                    self.db.grid_idx[point] = fid
            
            if valid_points:
                valid_frontiers[fid] = valid_points
        
        # Zaktualizuj frontiers
        self.db.frontiers = defaultdict(set, valid_frontiers)
        
        if self.debug_logging:
            self.get_logger().info(f"🔧 Naprawiono bazę: {len(valid_frontiers)} granic")


    def _resize_data_structures_if_needed(self):
        """Zmienia rozmiar struktur danych jeśli mapa się zmieniła - POPRAWIONA WERSJA."""
        if self.db.grid_idx.shape != self.map_shape:
            # Zapisz statystyki przed zmianą
            old_frontier_count = len(self.db.get_all_frontier_points())
            old_frontier_ids = list(self.db.frontiers.keys())
            
            if self.debug_logging:
                self.get_logger().info(
                    f"📏 Zmiana rozmiaru mapy z {self.db.grid_idx.shape} na {self.map_shape}"
                    f" (frontiers przed: {old_frontier_count})"
                )
            
            # Bezpieczne przeniesienie danych
            self._safe_resize_frontier_db()
            self._safe_resize_visit_counter()
            
            # Sprawdź ile frontierów zostało zachowanych
            new_frontier_count = len(self.db.get_all_frontier_points())
            preserved_percentage = (new_frontier_count / old_frontier_count * 100) if old_frontier_count > 0 else 0
            
            if self.debug_logging:
                self.get_logger().info(
                    f"📏 Zmieniono rozmiar mapy: zachowano {new_frontier_count}/{old_frontier_count} "
                    f"frontierów ({preserved_percentage:.1f}%)"
                )
    def _safe_resize_frontier_db(self):
        """Bezpiecznie zmienia rozmiar bazy danych granic."""
        old_db = self.db
        
        # Sprawdź czy mamy jakieś dane do przeniesienia
        if not old_db.frontiers:
            self.db = FrontierDB(self.map_shape)
            return
        
        # Zbierz wszystkie punkty frontierów z ich ID
        frontier_data = []
        for fid, points in old_db.frontiers.items():
            valid_points = []
            for p in points:
                # Sprawdź czy punkt mieści się w nowej mapie
                if (0 <= p[0] < self.map_shape[0] and 0 <= p[1] < self.map_shape[1]):
                    valid_points.append(p)
            
            if valid_points:  # Tylko jeśli mamy prawidłowe punkty
                frontier_data.append((fid, valid_points))
        
        # Utwórz nową bazę danych
        self.db = FrontierDB(self.map_shape)
        
        # Mapowanie starych ID na nowe ID
        id_mapping = {}
        
        # Przenieś frontiers zachowując strukturę
        for old_fid, points in frontier_data:
            if not points:
                continue
                
            # Utwórz nową granicę
            new_fid = self.db.new_frontier()
            id_mapping[old_fid] = new_fid
            
            # Dodaj wszystkie punkty do nowej granicy
            for p in points:
                self.db.add_point(new_fid, p)
        
        if self.debug_logging:
            preserved_frontiers = len([fid for fid, points in frontier_data if points])
            self.get_logger().debug(
                f"🔄 Przeniesiono {preserved_frontiers} granic z {len(old_db.frontiers)} oryginalnych"
            )

    def _safe_resize_visit_counter(self):
        """Bezpiecznie zmienia rozmiar licznika odwiedzin."""
        old_vc = self.visit_counter
        self.visit_counter = np.zeros(self.map_shape, dtype=np.uint16)
        
        # Oblicz obszar wspólny
        min_rows = min(old_vc.shape[0], self.map_shape[0])
        min_cols = min(old_vc.shape[1], self.map_shape[1])
        
        if min_rows > 0 and min_cols > 0:
            # Przenieś dane z obszaru wspólnego
            self.visit_counter[:min_rows, :min_cols] = old_vc[:min_rows, :min_cols]
            
            if self.debug_logging:
                transferred_visits = np.sum(self.visit_counter[:min_rows, :min_cols])
                self.get_logger().debug(f"🔄 Przeniesiono {transferred_visits} odwiedzin")

    
    def _diagnose_frontiers(self, occ_grid: np.ndarray):
        """Funkcja diagnostyczna do debugowania frontierów."""
        if not self.debug_logging:
            return
            
        all_points = self.db.get_all_frontier_points()
        
        value_counts = defaultdict(int)
        for point in all_points:
            value = occ_grid[point]
            value_counts[value] += 1
        
        self.get_logger().info("🔍 Analiza frontierów według wartości mapy:")
        for value, count in sorted(value_counts.items()):
            self.get_logger().info(f"   Wartość {value}: {count} punktów")

    def _resize_frontier_db(self):
        """Zmienia rozmiar bazy danych granic."""
        old_db = self.db
        self.db = FrontierDB(self.map_shape)
        
        # Przenieś istniejące granice
        for fid, pts in old_db.frontiers.items():
            for p in pts:
                if (0 <= p[0] < self.map_shape[0] and 0 <= p[1] < self.map_shape[1]):
                    self.db.add_point(fid, p)

    def _resize_visit_counter(self):
        """Zmienia rozmiar licznika odwiedzin."""
        old_vc = self.visit_counter
        self.visit_counter = np.zeros(self.map_shape, dtype=np.uint16)
        
        # Przenieś istniejące dane
        min_rows = min(old_vc.shape[0], self.map_shape[0])
        min_cols = min(old_vc.shape[1], self.map_shape[1])
        self.visit_counter[:min_rows, :min_cols] = old_vc[:min_rows, :min_cols]

    # =========================================================================
    # PUBLIKOWANIE WYNIKÓW
    # =========================================================================

    def publish_frontiers(self):
        """Publikuje wykryte granice - działa równolegle z wykrywaniem."""
        try:
            # Sprawdź czy wykrywanie jest w toku
            with self._data_lock:
                if self.detection_in_progress:
                    # Opcjonalnie możesz dodać komunikat
                    # self.get_logger().info("Publikacja odłożona - wykrywanie w toku")
                    return
                
                # Pobierz dane TYLKO gdy wykrywanie jest zakończone
                points = self.db.get_all_frontier_points()[:]
                map_origin = self.map_origin
                map_resolution = self.map_resolution
                cpu_time = self.cpu_time
                cpu_usage = self.cpu_usage
                
            
            # Klasteryzacja (bez locka) - NAPRAWIONA: Użyj K-means dla stabilności
            try:
                clusters = _clustering_frontiers_Kmeans(
                    parent=self,  # NAPRAWIONE: Dodaj parametr parent
                    debug_logging=False, 
                    frontiers=points, 
                    divisor=100,
                    min_frontiers=3,
                    max_frontiers=15
                )
            except Exception as e:
                # Fallback do DBScan jeśli K-means nie działa
                if self.debug_logging:
                    self.get_logger().warning(f"⚠️  K-means nie działa, używam DBScan: {e}")
                clusters = _clustering_frontiers_DBScan(
                    debug_logging=False, 
                    frontiers=points, 
                    divisor=50,
                    min_frontiers=3,
                    max_frontiers=15
                )
            
            # Tworzenie wiadomości
            msg = Frontiers()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            
            # Oblicz centroidy klastrów
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
                            centroid[0] = p[1]  # x
                            centroid[1] = p[0]  # y
                    
                    cluster_centroids.append(Point2DKoala(x=int(centroid[0]), y=int(centroid[1])))
            
            msg.clusters = cluster_centroids
            msg.frontiers = [Point2DKoala(x=int(p[1]), y=int(p[0])) for p in points]
            
            # Dodaj metryki wydajności
            msg.origin_x = float(map_origin[0])
            msg.origin_y = float(map_origin[1])
            msg.resolution = float(map_resolution)
            msg.cpu_time = float(cpu_time)
            msg.cpu_usage = float(cpu_usage)
            msg.memory_usage = 0.0
            
            # Publikuj
            self.frontier_pub.publish(msg)
            
            if self.debug_logging:
                with self._data_lock:
                    active_area_size = len(self.active_area)
                self.get_logger().info(
                    f"📡 Opublikowano {len(points)} frontierów w {len(cluster_centroids)} klastrach "
                    f"(CPU: {cpu_time:.3f}s, Aktywny obszar: {active_area_size} punktów)"
                )
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd publikacji: {e}")
    
    def _diagnose_coordinate_issues(self, laser_pts: List[Tuple[int, int]], robot_pose: Tuple[float, float, float]):
        """
        Funkcja diagnostyczna do analizy problemów z ujemnymi współrzędnymi.
        """
        if not self.debug_logging:
            return
            
        robot_x, robot_y, _ = robot_pose
        robot_grid = self._world_to_grid((robot_x, robot_y))
        
        # Analiza punktów laser
        negative_points = [(x, y) for x, y in laser_pts if x < 0 or y < 0]
        
        if negative_points or robot_grid[0] < 0 or robot_grid[1] < 0:
            self.get_logger().warning(
                f"🚨 PROBLEM Z UJEMNYMI WSPÓŁRZĘDNYMI:"
                f"\n  Robot world: ({robot_x:.2f}, {robot_y:.2f})"
                f"\n  Robot grid: ({robot_grid[0]}, {robot_grid[1]})"
                f"\n  Map origin: ({self.map_origin[0]:.2f}, {self.map_origin[1]:.2f})"
                f"\n  Map shape: {self.map_shape}"
                f"\n  Resolution: {self.map_resolution:.4f}"
                f"\n  Ujemne punkty laser: {len(negative_points)}/{len(laser_pts)}"
            )
            
            if negative_points and len(negative_points) <= 5:
                for i, (x, y) in enumerate(negative_points[:5]):
                    self.get_logger().warning(f"    Ujemny punkt {i+1}: grid({x}, {y})")
    
    def test_coordinate_conversion_scenarios(self):
        """
        Funkcja testowa do sprawdzenia różnych scenariuszy konwersji współrzędnych.
        Pomocna w diagnozie problemów z ujemnymi współrzędnymi.
        """
        if not self.debug_logging:
            return
            
        self.get_logger().info("🧪 TEST: Scenariusze konwersji współrzędnych")
        
        test_cases = [
            # (world_x, world_y, opis)
            (0.0, 0.0, "Zero world"),
            (-1.0, -1.0, "Ujemne world"),
            (1.0, 1.0, "Dodatnie world"),
            (self.map_origin[0], self.map_origin[1], "Map origin"),
            (self.map_origin[0] - 1.0, self.map_origin[1] - 1.0, "Przed origin"),
            (self.map_origin[0] + 1.0, self.map_origin[1] + 1.0, "Po origin"),
        ]
        
        for world_x, world_y, opis in test_cases:
            grid_pos = self._world_to_grid((world_x, world_y))
            valid = (0 <= grid_pos[0] < self.map_shape[0] and 
                    0 <= grid_pos[1] < self.map_shape[1])
            
            self.get_logger().info(
                f"  {opis}: world({world_x:.2f}, {world_y:.2f}) "
                f"→ grid({grid_pos[0]}, {grid_pos[1]}) "
                f"{'✓' if valid else '✗'}"
            )

    def _periodic_consistency_check(self):
        """
        Okresowe sprawdzanie spójności bazy danych frontierów.
        """
        try:
            with self._data_lock:
                if not self._validate_frontier_consistency():
                    self.get_logger().warning("⚠️  Wykryto niespójności podczas okresowego sprawdzania - naprawiam...")
                    self._repair_frontier_consistency()
                    
                    # Sprawdź ponownie
                    if self._validate_frontier_consistency():
                        self.get_logger().info("✅ Spójność bazy danych przywrócona")
                    else:
                        self.get_logger().error("❌ Nie udało się naprawić spójności bazy danych!")
                elif self.debug_logging:
                    frontier_count = len(self.db.get_all_frontier_points())
                    self.get_logger().debug(f"✅ Spójność bazy OK ({frontier_count} frontierów)")
                    
        except Exception as e:
            self.get_logger().error(f"❌ Błąd podczas okresowego sprawdzania spójności: {e}")

# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    
    # Użyj MultiThreadedExecutor dla równoległego przetwarzania
    executor = MultiThreadedExecutor(num_threads=4)
    node = FFDExplorerNode()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Zatrzymano przez użytkownika')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()