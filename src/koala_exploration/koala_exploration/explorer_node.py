#!/usr/bin/env python3
from __future__ import annotations
from calendar import c
import os
import math
import threading
import time
from typing import List, Tuple, Set, Optional, Dict
import sys

import numpy as np
if not hasattr(np, "float"):
    np.float = float  # type: ignore

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from rcl_interfaces.msg import ParameterDescriptor

from .utils.utils_explorer_node import ExplorerUtils
from .utils.benchmark_detector_node import ParallelFrontierBenchmark, BenchmarkAnalyzer

from slam_toolbox.srv import SerializePoseGraph 
from slam_toolbox.srv import SaveMap
from std_msgs.msg import String

from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

from koala_intefaces.msg import Frontiers


# =========================
#   Parametry konfiguracyjne
# =========================


# --- Parametry threading ---
MAX_FRONTIER_CACHE_AGE = 1.2         # Maksymalny wiek cache'u frontierów (sekundy)

# --- Parametry tworzenia mapy ---
MAP_SAVE_DIR       = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'maps', time.strftime("%Y-%m-%d_%H-%M-%S"))

# =========================
#   Interaktywny wykres frontierów
# =========================

class FrontierGraph:
    """
    Klasa do zarządzania wykresem liczby frontierów bez interfejsu graficznego.
    Przygotowuje wykres w pamięci i umożliwia zapis do pliku.
    """
    
    def __init__(self, parent_node, enable_visualization=True, debug_logging=True):
        """
        Inicjalizuje wykres frontierów.
        
        Args:
            parent_node: Węzeł ROS2 nadrzędny do logowania
            enable_visualization (bool): Włącza/wyłącza tworzenie wykresu
            debug_logging (bool): Włącza/wyłącza szczegółowe logowanie
        """
        self.parent_node = parent_node
        self.enable_visualization = enable_visualization
        self.debug_logging = debug_logging
        self._frontier_history = []
        self.figure = None
        self.axis = None
        
        if self.parent_node:
            self.parent_node.get_logger().info("FrontierGraph - tryb zapisu do pliku")
        
        self._initialize_plot()
    
    def _initialize_plot(self):
        """
        Inicjalizuje wykres matplotlib z backendem bez GUI (Agg).
        Konfiguruje podstawowe parametry wykresu jak tytuł, etykiety osi i siatkę.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Backend bez GUI
            
            import matplotlib.pyplot as plt
            
            self.figure, self.axis = plt.subplots(figsize=(12, 8))
            self.axis.set_title('Liczba Frontierów w Czasie Eksploracji', fontsize=16)
            self.axis.set_xlabel('Czas [s]', fontsize=12)
            self.axis.set_ylabel('Liczba Frontierów', fontsize=12)
            self.axis.grid(True, alpha=0.3)
            
            if self.parent_node:
                self.parent_node.get_logger().info("Wykres przygotowany do zapisu")
                
        except Exception as e:
            if self.parent_node:
                self.parent_node.get_logger().error(f"Błąd inicjalizacji wykresu: {e}")
            self.enable_visualization = False
    
    def add_data_point(self, num_frontiers: int, timestamp: float = None): # type: ignore
        """
        Dodaje nowy punkt danych do historii frontierów.
        
        Args:
            num_frontiers (int): Liczba wykrytych frontierów
            timestamp (float, optional): Znacznik czasowy. Jeśli None, użyje bieżącego czasu
        """
        if timestamp is None:
            timestamp = time.time()
        
        self._frontier_history.append({
            'timestamp': timestamp,
            'count': num_frontiers
        })
        
        # Ogranicz historię do ostatnich 1000 punktów aby uniknąć nadmiernego zużycia pamięci
        if len(self._frontier_history) > 1000:
            self._frontier_history = self._frontier_history[-1000:]
        
        if self.parent_node and self.debug_logging:
            self.parent_node.get_logger().info(f"Frontiers: {num_frontiers} (łącznie punktów: {len(self._frontier_history)})")
    
    def _update_plot(self):
        """
        Aktualizuje wykres w pamięci na podstawie zapisanej historii danych.
        Rysuje linię trendu, zaznacza aktualny punkt i dodaje statystyki tekstowe.
        Wykres nie jest wyświetlany, tylko przygotowywany do zapisu.
        """
        if not self.enable_visualization or not self.figure or not self.axis:
            return
            
        try:
            if len(self._frontier_history) < 2:
                return
                
            timestamps = [entry['timestamp'] for entry in self._frontier_history]
            counts = [entry['count'] for entry in self._frontier_history]
            
            # Oblicz czas relatywny od pierwszego pomiaru
            if timestamps:
                start_time = timestamps[0]
                relative_times = [(t - start_time) for t in timestamps]
            else:
                relative_times = timestamps
            
            # Wyczyść poprzedni wykres i narysuj nowy
            self.axis.clear()
            self.axis.plot(relative_times, counts, 'b-', linewidth=2, alpha=0.8, label='Liczba frontierów')
            self.axis.scatter(relative_times[-1:], counts[-1:], color='red', s=50, zorder=5, label='Aktualny')
            
            self.axis.set_title('Liczba Frontierów w Czasie Eksploracji', fontsize=16)
            self.axis.set_xlabel('Czas [s]', fontsize=12)
            self.axis.set_ylabel('Liczba Frontierów', fontsize=12)
            self.axis.grid(True, alpha=0.3)
            self.axis.legend()
            
            # Dodaj pole tekstowe ze statystykami
            if counts:
                avg_count = sum(counts) / len(counts)
                max_count = max(counts)
                min_count = min(counts)
                current_count = counts[-1]
                
                stats_text = (f'Aktualnie: {current_count}\n'
                             f'Maksimum: {max_count}\n'
                             f'Minimum: {min_count}\n'
                             f'Średnia: {avg_count:.1f}')
                
                self.axis.text(0.02, 0.98, stats_text, 
                             transform=self.axis.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                             fontsize=10)
            
            # Zapisz wykres do bufora pamięci bez wyświetlania
            import io
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.close()
            
        except Exception as e:
            if self.parent_node:
                self.parent_node.get_logger().error(f"Błąd aktualizacji wykresu: {e}")
    
    def save_plot(self, filename: str):
        """
        Zapisuje aktualny wykres do pliku PNG.
        
        Args:
            filename (str): Ścieżka do pliku docelowego
        """
        if not self.enable_visualization or not self.figure:
            if self.parent_node:
                self.parent_node.get_logger().warning("Brak wykresu do zapisania")
            return
            
        try:
            # Aktualizuj wykres przed zapisem
            self._update_plot()
            
            # Zapisz do pliku z wysoką rozdzielczością
            self.figure.savefig(filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
            
            if self.parent_node:
                self.parent_node.get_logger().info(f"Wykres zapisany: {filename}")
                
        except Exception as e:
            if self.parent_node:
                self.parent_node.get_logger().error(f"Błąd zapisywania wykresu: {e}")
    
    def close(self):
        """
        Zamyka i zwalnia zasoby wykresu matplotlib.
        Zapobiega wyciekom pamięci po zakończeniu eksploracji.
        """
        if self.figure:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None
            self.axis = None
    
    def get_current_count(self) -> int:
        """
        Zwraca aktualną liczbę frontierów z ostatniego pomiaru.
        
        Returns:
            int: Liczba frontierów lub 0 jeśli brak danych
        """
        if self._frontier_history:
            return self._frontier_history[-1]['count']
        return 0
    
    def get_history(self) -> List[dict]:
        """
        Zwraca kopię całej historii pomiarów frontierów.
        
        Returns:
            List[dict]: Lista słowników z kluczami 'timestamp' i 'count'
        """
        return self._frontier_history.copy()


# =========================
#   Thread-safe frontier cache
# =========================
class FrontierCache:
    """
    Klasa do bezpiecznego zarządzania cache'em frontierów w środowisku wielowątkowym.
    Przechowuje wykryte frontiers, klastry oraz parametry mapy z synchronizacją dostępu.
    """
    
    def __init__(self):
        """
        Inicjalizuje cache frontierów z domyślnymi wartościami.
        Tworzy mutex do synchronizacji dostępu między wątkami.
        """
        self._lock = threading.Lock()
        self._frontiers = []
        self._clusters = []
        self._num_of_elements_in_clusters = [] 
        self._map_params = (0.0, 0.0, 0.05)
        self._timestamp = 0.0
        self._processing = False
        self._cpu_time = 0.0
        

    def update(self, frontiers, clusters, map_params, cpu_time=0.0):
        """
        Aktualizuje cache nowymi danymi frontierów w sposób bezpieczny dla wątków.
        
        Args:
            frontiers: Lista wykrytych frontierów
            clusters: Lista klastrów frontierów
            map_params: Parametry mapy (origin_x, origin_y, resolution)
            cpu_time: Czas przetwarzania CPU (opcjonalny)
        """
        with self._lock:
            self._frontiers = frontiers.copy()
            self._clusters = [cluster.copy() for cluster in clusters]
            # Tworzymy słownik: klucz = współrzędne klastra (punkt), wartość = liczba wystąpień tego punktu w etykietach
            self._num_of_elements_in_clusters = [
                    (tuple(cluster[0]), len(cluster)-1) for cluster in clusters if cluster
                ]
            self._map_params = map_params
            self._timestamp = time.time()
            self._cpu_time = cpu_time
    def get_num_of_elements_in_clusters(self, coords: Tuple[int, int]) -> int:
        """
        Pobiera liczbę elementów w klastrze o podanych współrzędnych.

        Args:
            coords: Współrzędne klastra (r, c)

        Returns:
            Liczba elementów w klastrze o podanych współrzędnych lub 0 jeśli nie znaleziono.
        """
        with self._lock:
            for cluster_coords, count in getattr(self, '_num_of_elements_in_clusters', []):
                if cluster_coords == coords:
                    return count
        return 0
            
    def get(self) -> Tuple[List, List, Tuple, float]:
        """
        Pobiera aktualnie przechowywane dane z cache'u w sposób bezpieczny dla wątków.
        
        Returns:
            Tuple zawierająca: (frontiers, clusters, map_params, timestamp)
        """
        with self._lock:
            return (
                self._frontiers.copy(),
                [cluster.copy() for cluster in self._clusters],
                self._map_params,
                self._timestamp
            )
    
    def is_fresh(self, max_age: float = MAX_FRONTIER_CACHE_AGE) -> bool:
        """
        Sprawdza czy dane w cache są świeże (nie starsze niż max_age).
        
        Args:
            max_age: Maksymalny dopuszczalny wiek danych w sekundach
            
        Returns:
            True jeśli dane są świeże, False w przeciwnym razie
        """
        with self._lock:
            return (time.time() - self._timestamp) < max_age
    
    def set_processing(self, processing: bool):
        """
        Ustawia flagę przetwarzania danych.
        
        Args:
            processing: True jeśli dane są aktualnie przetwarzane, False w przeciwnym razie
        """
        with self._lock:
            self._processing = processing
            
    def is_processing(self) -> bool:
        """
        Sprawdza czy dane są aktualnie przetwarzane.
        
        Returns:
            True jeśli przetwarzanie jest w toku, False w przeciwnym razie
        """
        with self._lock:
            return self._processing
            

class FrontierExplorer(Node):
    """
    Główna klasa do eksploracji frontierów w środowisku ROS2.
    Implementuje autonomiczne wykrywanie i nawigację do frontierów mapy.
    """
    
    def __init__(self) -> None:
        """
        Inicjalizuje węzeł eksploracji frontierów.
        Deklaruje parametry, tworzy subskrypcje i publikacje, inicjalizuje nawigator.
        """
        super().__init__("frontier_explorer")

        # Definicja parametrów
        self.declare_parameter('CLUSTER_RADIUS', 0.5, ParameterDescriptor(description="Promień grupowania frontierów w metrach"))
        self.declare_parameter('INFO_RADIUS', 0.5, ParameterDescriptor(description="Promień liczenia nieznanych komórek (metry)"))
        self.declare_parameter('INFO_WEIGHT', 5.5, ParameterDescriptor(description="Waga information gain w funkcji score"))
        self.declare_parameter('DIST_WEIGHT', 2500.0, ParameterDescriptor(description="Waga odległości w funkcji score"))
        self.declare_parameter('BLACKLIST_RADIUS', 0.4, ParameterDescriptor(description="Promień blacklistowania wokół odwiedzonych frontierów (metry)"))
        self.declare_parameter('SCORE_THRESHOLD', 400.0, ParameterDescriptor(description="Minimalny score frontiera do akceptacji"))
        self.declare_parameter('TIMER_PERIOD', 1.0, ParameterDescriptor(description="Częstotliwość głównego timera eksploracji (sekundy)"))
        self.declare_parameter('VISUALIZATION_REFRESH_PERIOD', 0.2, ParameterDescriptor(description="Częstotliwość odświeżania wizualizacji (sekundy)"))
        self.declare_parameter('ENABLE_VISUALIZATION', True, ParameterDescriptor(description="Włącz/wyłącz markery w RViz"))
        self.declare_parameter('MARKER_LIFETIME', 30.0, ParameterDescriptor(description="Czas życia markerów w RViz (sekundy)"))
        self.declare_parameter('DEBUG_LOGGING', False, ParameterDescriptor(description="Włącz szczegółowe logi"))
        self.declare_parameter('MIN_SCORE_IMPROVEMENT', 5.0, ParameterDescriptor(description="Minimalna poprawa score dla zmiany celu"))
        self.declare_parameter('MAP_SAVE_DIR', MAP_SAVE_DIR, ParameterDescriptor(description="Katalog do zapisywania mapy"))
        self.declare_parameter('MAP_SAVE_ENABLED', True, ParameterDescriptor(description="Włącz zapisywanie mapy"))
        self.declare_parameter('MAP_FAILED_SAVE_THRESHOLD', 10, ParameterDescriptor(description="Liczba nieudanych prób przed zapisem mapy"))
        self.declare_parameter('ADAPTIVE_INFO_GAIN', 1.5, ParameterDescriptor(description="O ile ma się zwiększyć waga information gain, jeśli nie ma poprawy score"))        
        self.declare_parameter('USE_FFD_SOURCE', False, ParameterDescriptor(description="Używaj frontierów z węzła FFD, w przeciwnym razie używaj WFD"))
        self.declare_parameter('USE_WFD_SOURCE', True, ParameterDescriptor(description="Używaj frontierów z węzła WFD, w przeciwnym razie używaj FFD"))
        self.declare_parameter('ENABLE_BENCHMARK', False, ParameterDescriptor(description="Włącz benchmark FFD vs WFD"))
        
                # =========================
        # BLOK: Inicjalizacja eksploratorów i benchmarku
        # =========================
        # Załaduj parametry JEDNORAZOWO na początku
        self.utils = ExplorerUtils(self)
        self.load_initial_parameters()
        self.get_logger().info("Parametry eksploracji załadowane")
        self.MAP_SAVE_DIR = MAP_SAVE_DIR
         
        # =========================
        # BLOK: Przygotowanie katalogów i logowanie
        # =========================
        if self.MAP_SAVE_DIR and not os.path.exists(self.MAP_SAVE_DIR):
            os.makedirs(self.MAP_SAVE_DIR)
            self.get_logger().info(f"Utworzono katalog do zapisywania map: {self.MAP_SAVE_DIR}")
        
        # Pokaż aktualną metodę
        method_name = "FFD (Fast Frontier Detector)" if getattr(self, 'ENABLE_FFD', True) else "WFD (Wavefront Frontier Detector)"
        self.get_logger().info(f"Metoda wykrywania frontierów: {method_name}")
        
        # =========================
        # BLOK: Inicjalizacja wykresu i nawigatora
        # =========================
        self.graph = FrontierGraph(
            parent_node=self, 
            enable_visualization=self.ENABLE_VISUALIZATION, 
            debug_logging=self.DEBUG_LOGGING
        )
        
        self._nav2_ready = False  # Flaga do sprawdzenia gotowości Nav2 Commander
        
        # Inicjalizacja nawigatora Nav2 Commander
        self.nav = BasicNavigator()
        self._check_nav2_ready()
        self.get_logger().info("Czekam na Nav2 Commander...")
        
        if self.DEBUG_LOGGING:
            self.get_logger().info("Debug logging włączone")
            self.get_logger().info(f"Parametry: {self.get_parameters(['TIMER_PERIOD'])}")
        
        # =========================
        # BLOK: Monitor postępu eksploracji
        # =========================
        self._progress_monitor = {
            'navigation_start_time': None,
            'initial_frontier_count': 0,
            'last_frontier_count': 0,
            'last_check_time': None,
            'total_reduction': 0,
            'CLUSTER_RADIUS': self.CLUSTER_RADIUS,
            'INFO_RADIUS': self.INFO_RADIUS,
            'INFO_WEIGHT': self.INFO_WEIGHT,
            'DIST_WEIGHT': self.DIST_WEIGHT,
            'BLACKLIST_RADIUS': self.BLACKLIST_RADIUS,
            'SCORE_THRESHOLD': self.SCORE_THRESHOLD,
            'TIMER_PERIOD': self.TIMER_PERIOD,
            'MIN_SCORE_IMPROVEMENT': self.MIN_SCORE_IMPROVEMENT,
            'MAP_FAILED_SAVE_THRESHOLD': self.MAP_FAILED_SAVE_THRESHOLD,
            'ADAPTIVE_INFO_GAIN': self.ADAPTIVE_INFO_GAIN,
            'ENABLE_FFD': self.USE_FFD_SOURCE
        }
        
        # =========================
        # BLOK: Subskrypcje i publikatory ROS2
        # =========================
        # Subskrypcje
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)
        
        # Subskrypcje do frontierów z zewnętrznych węzłów
        self.wfd_frontiers_sub = self.create_subscription(
            Frontiers, 
            'WFD/frontiers', 
            self._wfd_frontiers_callback, 
            10
        )
        
        self.ffd_frontiers_sub = self.create_subscription(
            Frontiers, 
            'FFD/frontiers', 
            self._ffd_frontiers_callback, 
            10
        )
        
        self._setup_odometry_subscription()
        
        try:
            if self.ENABLE_BENCHMARK:
                self.benchmark = ParallelFrontierBenchmark(
                    parent_node=self,
                    max_results=5000
                )
                self.analyzer = BenchmarkAnalyzer(self.benchmark)
                self.get_logger().info("Benchmark równoległy zainicjalizowany")
                
                # Timer do raportowania wyników
                self.benchmark_timer = self.create_timer(10.0, self._report_benchmark_results)
            else:
                self.benchmark = None
        except Exception as e:
            self.get_logger().error(f"Błąd inicjalizacji benchmarku: {e}")
            self.benchmark = None
        
        # Publikacje dla wizualizacji
        if self.ENABLE_VISUALIZATION:
            self.marker_pub = self.create_publisher(MarkerArray, "/frontier_markers", 10)
            self.blacklist_pub = self.create_publisher(MarkerArray, "/blacklist_markers", 10)
            self.info_pub = self.create_publisher(MarkerArray, "/frontier_info_markers", 10)
            
        # =========================
        # BLOK: Inicjalizacja transformacji TF
        # =========================
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        
        # =========================
        # BLOK: Zmienne stanu nawigacji
        # =========================
        # Stan nawigacji
        self._map: Optional[OccupancyGrid] = None
        self._frontier_blacklist: Set[Tuple[int,int]] = set()
        self._current_target: Optional[Tuple[int,int]] = None
        self._current_target_score: float = -math.inf
        self._previous_target = None  # Poprzedni cel do nawigacji
        self._visited_positions = []  # Lista odwiedzonych pozycji robota
        self._map_saved = False
        
        self._last_goal_sent = None
        self._last_navigation_attempt = 0.0
        self._last_continuous_eval = 0.0
        
        self._frontier_attempt_history: Dict[Tuple[int,int], int] = {}  # Liczba prób na frontier
        self._max_attempts_per_frontier = 2  # Maksymalnie 2 próby na frontier
        
        # Inicjalizacja zmiennych odometrii
        self._last_odom_msg = None
        self._last_odom_log_time = 0.0
        
        # =========================
        # BLOK: Tracking wydajności CPU
        # =========================
        self._total_cpu_time = 0.0
        self._detection_count = 0
        
        # Thread-safe frontier cache
        self._frontier_cache = FrontierCache()
        
        # =========================
        # BLOK: Tracking pozycji robota dla blacklistowania
        # =========================
        self._robot_position_history = []
        self._total_distance_traveled = 0.0
        self._last_blacklist_position = None
        self._blacklist_after_distance = 5.0  # Rozpocznij blacklistowanie po 5m
        self._position_blacklist_interval = 0.2  # Blacklistuj co 0.5m
        self._last_position_for_distance = None
        
        # =========================
        # BLOK: Zarządzanie wątkami
        # =========================
        # Threading
        self._frontier_thread = None
        self._shutdown_event = threading.Event()
        self._map_queue = []
        self._map_queue_lock = threading.Lock()
        
        # =========================
        # BLOK: Liczniki błędów i czas eksploracji
        # =========================
        # Licznik nieudanych prób znalezienia frontiera z wymaganym score
        self._failed_frontier_attempts = 0
        self._max_failed_attempts =  self.MAP_FAILED_SAVE_THRESHOLD # Po MAP_FAILED_SAVE_THRESHOLD nieudanych próbach zapisz mapę
        
        # Czas eksploracji
        self.exploring_time = 0.0  # Czas eksploracji w sekundach
        self._cluster_route_optimized = False
        self._optimal_cluster_order = []
        self._current_cluster_index = 0
        self._last_cluster_count = 0 
        
        # =========================
        # BLOK: Timery ROS2
        # =========================
        # Timery
        self.timer_params = self.create_timer(5.0, self.utils.timer_callback)
        if self.DEBUG_LOGGING:
            self.get_logger().info(f"Timer został poprawnie ustawiony {True if self.timer_params is not None else False}")

        # Timer do odświeżania wizualizacji frontierów
        if self.ENABLE_VISUALIZATION:
            self._viz_timer = self.create_timer(self.VISUALIZATION_REFRESH_PERIOD, self._refresh_visualization)
        
        # =========================
        # BLOK: Publisher kontroli robota
        # =========================
        # Publisher do sterowania robotem
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        if self.DEBUG_LOGGING:
            self.get_logger().info(f"Publisher cmd_vel został poprawnie ustawiony subskrybenci: {True if self.cmd_vel_pub is not None else False}")

        # Czas eksploracji
        self.exploring_time = 0.0  # Czas eksploracji w sekundach
        self.attempt_blacklist = 0
        
        # =========================
        # BLOK: Statystyki eksploracji
        # =========================
        # Tracking statystyk eksploracji
        self.exploration_stats = {
            'start_time': None,
            'end_time': None,
            'total_distance': 0.0,
            'path_points': [],
            'last_position': None,
            'frontiers_visited': 0,
            'navigation_attempts': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'frontiers_not_visited': 0,
        }
        
        # =========================
        # BLOK: Timer trackingu pozycji
        # =========================
        # Timer do trackingu pozycji (co sekundę)
        self.position_timer = self.create_timer(1.0, self._track_position)
    
    def _setup_odometry_subscription(self) -> None:
        """
        Tworzy subskrypcję na topic /odometry/filtered.
        Inicjalizuje zmienne do przechowywania danych odometrii.
        Powinno być wywołane w __init__.
        """
        
        # Tworzenie subskrypcji
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self._odom_callback,
            10
        )
        
        self.get_logger().info("Subskrypcja na /odometry/filtered utworzona")
    
    def load_initial_parameters(self) -> None:
        """Ładuje początkowe wartości parametrów jako atrybuty klasy"""
        self.CLUSTER_RADIUS = self.get_parameter('CLUSTER_RADIUS').get_parameter_value().double_value
        self.INFO_RADIUS = self.get_parameter('INFO_RADIUS').get_parameter_value().double_value
        self.INFO_WEIGHT = self.get_parameter('INFO_WEIGHT').get_parameter_value().double_value
        self.DIST_WEIGHT = self.get_parameter('DIST_WEIGHT').get_parameter_value().double_value
        self.BLACKLIST_RADIUS = self.get_parameter('BLACKLIST_RADIUS').get_parameter_value().double_value
        self.SCORE_THRESHOLD = self.get_parameter('SCORE_THRESHOLD').get_parameter_value().double_value
        self.TIMER_PERIOD = self.get_parameter('TIMER_PERIOD').get_parameter_value().double_value
        self.VISUALIZATION_REFRESH_PERIOD = self.get_parameter('VISUALIZATION_REFRESH_PERIOD').get_parameter_value().double_value
        self.ENABLE_VISUALIZATION = self.get_parameter('ENABLE_VISUALIZATION').get_parameter_value().bool_value
        self.MARKER_LIFETIME = self.get_parameter('MARKER_LIFETIME').get_parameter_value().double_value
        self.DEBUG_LOGGING = self.get_parameter('DEBUG_LOGGING').get_parameter_value().bool_value
        self.MIN_SCORE_IMPROVEMENT = self.get_parameter('MIN_SCORE_IMPROVEMENT').get_parameter_value().double_value
        self.MAP_SAVE_DIR = self.get_parameter('MAP_SAVE_DIR').get_parameter_value().string_value
        self.MAP_SAVE_ENABLED = self.get_parameter('MAP_SAVE_ENABLED').get_parameter_value().bool_value
        self.MAP_FAILED_SAVE_THRESHOLD = self.get_parameter('MAP_FAILED_SAVE_THRESHOLD').get_parameter_value().integer_value
        self.ADAPTIVE_INFO_GAIN = self.get_parameter('ADAPTIVE_INFO_GAIN').get_parameter_value().double_value
        self.ENABLE_BENCHMARK = self.get_parameter('ENABLE_BENCHMARK').get_parameter_value().bool_value
        self.USE_FFD_SOURCE = self.get_parameter('USE_FFD_SOURCE').get_parameter_value().bool_value
        self.USE_WFD_SOURCE = self.get_parameter('USE_WFD_SOURCE').get_parameter_value().bool_value
        
    def _wfd_frontiers_callback(self, msg: Frontiers) -> None:
        """
        Callback do obsługi wiadomości z frontierami z węzła WFD.
        
        Args:
            msg: Wiadomość zawierająca listę frontierów
        """
        if not self._nav2_ready:
            self.get_logger().warning("Nav2 nie jest gotowy, ignoruję wiadomość z frontierami WFD")
            return
            
        if self.USE_WFD_SOURCE:  # Tylko jeśli używamy WFD jako źródła
            self._process_frontiers_message(msg, source="WFD")

    def _ffd_frontiers_callback(self, msg: Frontiers) -> None:
        """
        Callback do obsługi wiadomości z frontierami z węzła FFD.
        
        Args:
            msg: Wiadomość zawierająca listę frontierów
        """
        if not self._nav2_ready:
            self.get_logger().warning("Nav2 nie jest gotowy, ignoruję wiadomość z frontierami FFD")
            return
            
        if self.USE_FFD_SOURCE:  # Tylko jeśli używamy FFD jako źródła
            self._process_frontiers_message(msg, source="FFD")
            

    def _process_frontiers_message(self, msg: Frontiers, source: str) -> None:
        """
        Przetwarza wiadomość z frontierami z dowolnego źródła i aktualizuje cache.
        Args:
            msg: Wiadomość Frontiers
            source: Źródło wiadomości ("FFD" lub "WFD")
        """
        try:
            if self._frontier_cache.is_fresh():
                msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                cached_timestamp = self._frontier_cache.get()[3]
                
                # Jeśli dane w cache są nowsze, ignoruj tę wiadomość
                if cached_timestamp > msg_timestamp:
                    if self.DEBUG_LOGGING:
                        self.get_logger().debug(f"Ignoruję starsze dane z {source}, timestamp: {msg_timestamp:.2f} < {cached_timestamp:.2f}")
                    return
            
            # 1) Konwersja frontierów
            frontiers: List[Tuple[int,int]] = [(pt.y, pt.x) for pt in msg.frontiers]
            # 2) Wyciągnięcie centroidów
            centroids: List[Tuple[int,int]] = [(pt.y, pt.x) for pt in msg.clusters]
            # 3) Grupowanie: pierwszy element = centroid, potem frontier points
            clusters: List[List[Tuple[int,int]]] = []
            if centroids:
                clusters_map: Dict[Tuple[int,int], List[Tuple[int,int]]] = {c: [] for c in centroids}
                for f in frontiers:
                    nearest = min(centroids,
                                key=lambda c: (c[0]-f[0])**2 + (c[1]-f[1])**2)
                    clusters_map[nearest].append(f)
                clusters = [[c] + pts for c, pts in clusters_map.items()]
            # 4) Aktualizacja cache
            self._frontier_cache.update(
                frontiers=frontiers,
                clusters=clusters,
                map_params=(msg.origin_x, msg.origin_y, msg.resolution),
                cpu_time=msg.cpu_time
            )
            # 5) Wykres
            self.graph.add_data_point(len(frontiers))

            if self.DEBUG_LOGGING:
                self.get_logger().info(
                    f"{source}: Otrzymano {len(frontiers)} frontierów i {len(clusters)} klastrów, "
                    f"CPU time: {msg.cpu_time*1000:.2f}ms"
                )
        except Exception as e:
            self.get_logger().error(f"Błąd przetwarzania wiadomości z frontierami {source}: {e}")
            return
    # Modyfikacja metody start() - nie trzeba uruchamiać wątku detekcji, tylko timer główny
    def start(self):
        """
        Uruchamia główną pętlę eksploracji.
        Inicjalizuje parametry nawigacji i tworzy timer główny.
        """
        # Parametry nawigacji
        self.exploration_stats['navigation_parameters'] = self.utils.get_nav2_parameters()
        
        self._timer = self.create_timer(self.TIMER_PERIOD, self._explore)
        if self.DEBUG_LOGGING:
            self.get_logger().info(f"Timer został poprawnie ustawiony {self._timer.is_ready()}")
        
        
        # Zapisz czas startu eksploracji
        self.exploration_stats['start_time'] = self.get_current_time()
        self.exploring_time_start = self.nav.get_clock().now().seconds_nanoseconds()[0]
        
        # Uruchom benchmark tylko jeśli włączony
        if self.ENABLE_BENCHMARK and self.benchmark:
            self.benchmark.start_benchmark()
            self.get_logger().info("Benchmark uruchomiony - porównuję FFD vs WFD")
    
    def _track_position(self) -> None:
        """
        Callback timera do śledzenia pozycji robota w czasie.
        Zapisuje punkty trasy i oblicza całkowitą przebytą odległość.
        NOWE: Dodaje ciągłe blacklistowanie pozycji robota po przejechaniu 5m.
        """
        pose = self._get_robot_pose()
        if not pose:
            self.get_logger().warning("Brak pozycji robota, nie mogę śledzić trasy")
            return
            
        current_pos = (pose[0], pose[1])
        current_time = self.get_current_time()
        
        self._last_position_for_distance = current_pos
        
        # Dodaj punkt do trasy
        self.exploration_stats['path_points'].append({
            'x': pose[0],
            'y': pose[1],
            'timestamp': current_time
        })
        
        # Oblicz dystans od ostatniej pozycji (dla statystyk)
        if self.exploration_stats['last_position']:
            last_x, last_y = self.exploration_stats['last_position']
            distance = math.sqrt((pose[0] - last_x)**2 + (pose[1] - last_y)**2)
            self.exploration_stats['total_distance'] += distance
            
        self.exploration_stats['last_position'] = current_pos  

        # -------- Callbacks --------
    # Modyfikacja metody _map_callback - nie dodajemy do kolejki, tylko ewentualnie do benchmarku
    def _map_callback(self, msg: OccupancyGrid) -> None:
        """
        Callback dla nowych danych mapy - zapisuje mapę dla celów benchmarku.
        """
        self._map = msg
        
        # Sprawdź czy benchmark istnieje przed użyciem
        if hasattr(self, 'benchmark') and self.benchmark:
            self.benchmark.add_map_for_benchmark(msg)
            
    def _odom_callback(self, msg) -> None:
        """
        Callback dla topicu /odometry/filtered.
        Zapisuje ostatnią wiadomość odometrii do użycia przez inne funkcje.
        
        Args:
            msg: Wiadomość nav_msgs/Odometry
        """
        self._last_odom_msg = msg
        
        if hasattr(self, '_last_odom_log_time'):
            current_time = self.get_current_time()
            if current_time - self._last_odom_log_time > 0.1:  # Log co 0.1 sekundy
                self._last_odom_log_time = current_time
        else:
            self._last_odom_log_time = self.get_current_time()
        
    def _get_robot_pose(self) -> Optional[Tuple[float, float]]:
        """
        Pobiera pozycję robota (x, y) z topicu /odometry/filtered.
        
        Returns:
            Tuple (x, y) lub None jeśli dane niedostępne
        """
        try:
            # Sprawdź czy mamy zapisaną ostatnią odometrię
            if hasattr(self, '_last_odom_msg') and self._last_odom_msg:
                odom = self._last_odom_msg
                return odom.pose.pose.position.x, odom.pose.pose.position.y
            else:
                if self.DEBUG_LOGGING:
                    self.get_logger().warning("Brak danych odometrii")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Błąd pobierania pozycji z odometrii: {e}")
            return None

    def _get_robot_pose_full(self) -> Optional[Tuple[float, float, float]]:
        """
        Pobiera pełną pozę robota (x, y, yaw) z topicu /odometry/filtered.
        Konwertuje quaternion na kąt yaw.
        
        Returns:
            Tuple (x, y, yaw) lub None jeśli dane niedostępne
        """
        try:
            if hasattr(self, '_last_odom_msg') and self._last_odom_msg:
                odom = self._last_odom_msg
                
                x = odom.pose.pose.position.x
                y = odom.pose.pose.position.y
                
                # Konwertuj quaternion na yaw
                q = odom.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                
                return x, y, yaw
            else:
                if self.DEBUG_LOGGING:
                    self.get_logger().warning("Brak danych odometrii dla pełnej pozy")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Błąd pobierania pełnej pozy z odometrii: {e}")
            return None   
    
    def _check_nav2_ready(self) -> None:
        """
        Sprawdza czy Nav2 jest gotowy z kontrolowaną częstotliwością.
        Używa timera ROS2 zamiast pętli while dla lepszej wydajności.
        """
        # Użyj timera ROS2 zamiast pętli while
        self._nav2_check_timer = self.create_timer(0.1, self._check_nav2_status)
    
    def _report_benchmark_results(self):
        if not self.ENABLE_BENCHMARK or not self.benchmark:
            return

        comparison = self.benchmark.get_performance_comparison()
        if not comparison:
            return

        current_method = "FFD" if getattr(self, 'ENABLE_FFD', True) else "WFD"

        self.get_logger().info("=== BENCHMARK REPORT ===")
        self.get_logger().info(f"Aktualna metoda: {current_method}")
        ffd_time = comparison.get('ffd_avg_time', 0.0)
        wfd_time = comparison.get('wfd_avg_time', 0.0)
        self.get_logger().info(f"FFD avg time: {ffd_time*1000:.2f}ms")
        self.get_logger().info(f"WFD avg time: {wfd_time*1000:.2f}ms")
        
        try:
            if wfd_time == 0:
                speedup = float('inf') if ffd_time > 0 else 0
            else:
                speedup = ffd_time / wfd_time
            self.get_logger().info(f"Speedup ratio: {speedup:.2f}x")
        except Exception as e:
            self.get_logger().error(f"Error calculating speedup ratio: {e}")
            speedup = 0.0

        self.get_logger().info(f"FFD avg frontiers: {comparison.get('ffd_avg_frontiers', 0):.1f}")
        self.get_logger().info(f"WFD avg frontiers: {comparison.get('wfd_avg_frontiers', 0):.1f}")
        self.get_logger().info(f"Samples: FFD={comparison.get('ffd_samples', 0)}, WFD={comparison.get('wfd_samples', 0)}")

        # Określ zwycięzcę
        winner = "Brak porównywalnych danych!"
        if speedup > 1.0:
            winner = f"FFD jest {speedup:.2f}x szybszy!"
        elif 0 < speedup <= 1.0:
            winner = f"WFD jest {1/speedup:.2f}x szybszy!"
        self.get_logger().info(f"{winner}")
        
    def get_current_time(self) -> float:
        """Zwraca aktualny czas w sekundach od epoki."""
        return self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
    
    def _check_nav2_status(self) -> None:
        """
        Timer callback sprawdzający status Nav2 poprzez serwis velocity_smoother.
        Wywołuje asynchroniczne żądanie i ustawia callback dla odpowiedzi.
        """
        try:
            from lifecycle_msgs.srv import GetState
            
            # Utwórz klienta serwisu
            client = self.create_client(GetState, '/velocity_smoother/get_state')
            
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warning("Serwis /velocity_smoother/get_state niedostępny")
                return  # Timer automatycznie spróbuje ponownie za 1s
            
            # Wyślij żądanie asynchronicznie
            request = GetState.Request()
            future = client.call_async(request)
            
            # Dodaj callback do future
            future.add_done_callback(self._handle_nav2_response)
            
        except Exception as e:
            self.get_logger().error(f"Błąd sprawdzania velocity_smoother: {e}")
    
    def _handle_nav2_response(self, future) -> None:
        """
        Obsługuje odpowiedź od serwisu velocity_smoother.
        Sprawdza czy Nav2 jest w stanie aktywnym i uruchamia eksplorację.
        """
        try:
            response = future.result()
            if response:
                state_label = response.current_state.label
                state_id = response.current_state.id
                
                self.get_logger().info(f"velocity_smoother state: {state_label} (id: {state_id})")
                
                # Sprawdź czy jest w stanie aktywnym
                if state_id == 3:
                    # Nav2 gotowy - zatrzymaj timer i uruchom eksplorację
                    self._nav2_check_timer.cancel()
                    self._nav2_ready = True                   
                    self.get_logger().info("Nav2 gotowy - uruchamiam eksplorację")
                    self.start()  # Uruchom główną logikę eksploracji
                    return
                else:
                    self.get_logger().warning(f"velocity_smoother nie jest gotowy - stan: {state_label} (id: {state_id})")
                    # Timer automatycznie sprawdzi ponownie za 1 sekundę
            else:
                self.get_logger().error("Brak odpowiedzi od velocity_smoother")
                
        except Exception as e:
            self.get_logger().error(f"Błąd obsługi odpowiedzi Nav2: {e}")
    
    # ------ Nawigacja z Nav2 Commander API ------
    def _send_goal(self, x: float, y: float, cell: Tuple[int,int], score: float = 0.0, retry_attempt: bool = False) -> None:
        """
        Wysyła cel nawigacji z dynamicznym dostosowaniem tolerancji celu.
        Wykonuje walidację parametrów, sprawdza granice mapy i wykonalność ścieżki.
        
        Args:
            x, y: Współrzędne celu w systemie map
            cell: Komórka frontiera na mapie dyskretnej
            score: Score frontiera
            retry_attempt: Czy to próba ponowienia nawigacji
        """
        current_time = self.get_current_time()
        self.get_logger().info(f"_send_goal() START [{current_time:.1f}]: cel=({x:.2f}, {y:.2f}), cell={cell}, score={score:.2f}")
        
        try:
            # BLOK: Sprawdzenie stanu nawigatora
            if not hasattr(self, 'nav') or self.nav is None:
                self.get_logger().error("Navigator nie jest zainicjalizowany!")
                return
            
            # BLOK: Sprawdzenie gotowości Nav2
            try:
                test_pose = self._get_robot_pose()
                if not test_pose:
                    self.get_logger().error("Nie można uzyskać pozycji robota - Nav2 może nie być gotowy")
                    return
            except Exception as e:
                self.get_logger().error(f"Problem z Nav2: {e}")
                return
            
            # BLOK: Walidacja podstawowych parametrów
            if x is None or y is None or cell is None:
                self.get_logger().error(f"Nieprawidłowe parametry celu: x={x}, y={y}, cell={cell}")
                return
            
            # BLOK: Sprawdzenie granic mapy
            if not self._is_goal_within_map_bounds(x, y):
                self.get_logger().error(f"Cel ({x:.3f}, {y:.3f}) poza granicami mapy - ODRZUCAM")
                return
            
            # BLOK: Sprawdzenie istnienia celu w aktualnych frontierach
            _, clusters, _, is_fresh = self._get_cached_frontiers()
            if is_fresh and not self._is_target_cell_valid(cell, clusters):
                self.get_logger().warning(f"Próba wysłania nieistniejącego celu {cell} - IGNORUJĘ")
                return
            
            
            # BLOK: Ochrona przed duplikacją celów
            if not retry_attempt:
                if hasattr(self, '_last_goal_sent') and self._last_goal_sent:
                    last_x, last_y, last_cell = self._last_goal_sent
                    distance_to_last = math.hypot(x - last_x, y - last_y)
                    if distance_to_last < 0.1 and cell == last_cell:
                        self.get_logger().warning(f"Próba wysłania tego samego celu! ({x:.2f}, {y:.2f}) - IGNORUJĘ")
                        return
                
                if self._current_target == cell:
                    self.get_logger().warning(f"Już nawigujemy do celu {cell} - IGNORUJĘ")
                    return
            
            # BLOK: Anulowanie poprzedniej nawigacji
            self.get_logger().info(f"Anulowanie poprzedniej nawigacji do {self._current_target}")
            self.nav.cancelTask()
        
            # BLOK: Przygotowanie goal pose
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.position.z = 0.0
            goal_pose.pose.orientation.w = 1.0
    
            # BLOK: Zapisanie stanu nawigacji
            self._last_goal_sent = (x, y, cell)
            old_target = self._current_target
            
            # Zapisz poprzedni cel przed ustawieniem nowego
            if self._current_target is not None:
                self._previous_target = self._current_target
                self.get_logger().info(f"Zapisuję poprzedni cel: {self._previous_target}")
            
            self._current_target = cell
            self._current_target_score = score
            self.exploration_stats['navigation_attempts'] += 1
            
            # BLOK: Wysłanie celu do Nav2
            self.get_logger().info(f"Wysyłam cel do Nav2: ({x:.2f}, {y:.2f})")
            self.nav.goToPose(goal_pose)
            
            # BLOK: Logowanie sukcesu
            retry_text = " (RETRY)" if retry_attempt else ""
            change_text = f" (ZMIANA: {old_target}→{cell})" if old_target != cell and old_target is not None else ""
            
            self.get_logger().info(
                f"CEL WYSŁANY{retry_text}{change_text}: "
                f"({x:.2f}, {y:.2f}), score: {score:.2f}, cell: {cell}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Błąd wysyłania celu: {e}")
            # Reset stanu przy błędzie
            self._current_target = None
            self._current_target_score = -math.inf           
        
    def _is_goal_within_map_bounds(self, x: float, y: float) -> bool:
        """
        Sprawdza czy cel jest w granicach mapy z marginesem bezpieczeństwa.
        
        Args:
            x, y: Współrzędne celu w systemie map
            
        Returns:
            True jeśli cel jest w bezpiecznych granicach mapy, False w przeciwnym razie
        """
        if not self._map:
            return False
        
        try:
            # Pobierz parametry mapy
            grid, origin_x, origin_y, resolution = self._map_to_numpy(self._map)
            rows, cols = grid.shape
            
            # Konwertuj cel na komórkę mapy
            goal_c = int((x - origin_x) / resolution)
            goal_r = int((y - origin_y) / resolution)
            
            # Sprawdź z marginesem bezpieczeństwa
            safety_margin = 3
            
            is_within = (safety_margin <= goal_r < rows - safety_margin and 
                        safety_margin <= goal_c < cols - safety_margin)
            
            if not is_within and self.DEBUG_LOGGING:
                self.get_logger().warning(
                    f"Cel poza granicami: ({x:.3f},{y:.3f}) -> cell({goal_r},{goal_c}), "
                    f"map_size=({rows},{cols}), margin={safety_margin}"
                )
            
            return is_within
            
        except Exception as e:
            self.get_logger().error(f"Błąd sprawdzania granic mapy: {e}")
            return False
    
    def _is_target_cell_valid(self, target_cell: Tuple[int,int], clusters: List[List[Tuple[int,int]]]) -> bool:
        """
        Sprawdza czy dana komórka nadal istnieje w klastrach frontierów.
        
        Args:
            target_cell: Komórka do sprawdzenia (r, c)
            clusters: Lista klastrów frontierów
            
        Returns:
            True jeśli komórka została znaleziona w którymkolwiek klastrze
        """
        for cluster in clusters:
            if target_cell in cluster:
                return True
        return False
    
    def _send_goal_fallback(self, x: float, y: float, cell: Tuple[int,int], score: float) -> None:
        """
        Fallback dla _send_goal z obsługą błędów.
        Próbuje wysłać cel i w przypadku błędu resetuje stan nawigacji.
        """
        try:
            self._send_goal(x, y, cell, score)
        except Exception as e:
            self.get_logger().error(f"Błąd fallback nawigacji: {e}")
            # Reset stanu i znajdź inny cel
            self._current_target = None
            self._current_target_score = -math.inf
    
    def _get_cached_frontiers(self) -> Tuple[List, List, Tuple, bool]:
        """
        Pobiera frontiers z cache'u jeśli są świeże.
        
        Returns:
            Tuple zawierająca: (frontiers, clusters, map_params, is_fresh)
        """
        frontiers, clusters, map_params, _ = self._frontier_cache.get()
        is_fresh = self._frontier_cache.is_fresh()
        
        if is_fresh and frontiers is not None:
            return frontiers, clusters, map_params, True
        else:
            return [], [], (0.0, 0.0, 0.05), False
    
    def _refresh_visualization(self) -> None:
        """
        Timer callback do odświeżania wizualizacji frontierów i markersów.
        Wizualizuje okrąg lokalnej eksploracji oraz frontiers z cache'u.
        """
        if not self.ENABLE_VISUALIZATION:
            return
            
        # Użyj cache'u frontierów do wizualizacji
        frontiers, clusters, map_params, is_fresh = self._get_cached_frontiers()
        
        if is_fresh and frontiers:
            ox, oy, res = map_params
            self.utils.visualize_frontiers(frontiers, clusters, ox, oy, res, self._frontier_blacklist, self.marker_pub, self._cell_to_world)
            self.utils.visualize_blacklist(self._frontier_blacklist, ox, oy, res, self.blacklist_pub, self._cell_to_world)
    
    def _is_target_still_valid(self, clusters: List[List[Tuple[int,int]]]) -> bool:
        """
        Sprawdza czy aktualny cel nadal istnieje w dostępnych klastrach.
        Implementuje dodatkową tolerancję dla frontierów w pobliżu oryginalnego celu.
        """
        try:
            if not self._current_target:
                return False
            
            # Sprawdź czy aktualny cel jest w którymkolwiek z klastrów
            for cluster in clusters:
                if self._current_target in cluster:
                    if self.DEBUG_LOGGING:
                        self.get_logger().info(f"Cel {self._current_target} nadal istnieje w klastrze")
                    return True
            
            # Sprawdź czy cel nie jest na blackliście (dodatkowa ochrona)
            if self._current_target in self._frontier_blacklist:
                if self.DEBUG_LOGGING:
                    self.get_logger().info(f"Cel {self._current_target} jest na blackliście")
                return False
            
            # BLOK: Dodatkowe sprawdzenie - czy w okolicy celu są jeszcze frontiers
            tolerance_cells = 5  # 5 komórek tolerancji
            target_r, target_c = self._current_target
            
            for cluster in clusters:
                for frontier_cell in cluster:
                    fr, fc = frontier_cell
                    distance = math.sqrt((fr - target_r)**2 + (fc - target_c)**2)
                    if distance <= tolerance_cells:
                        if self.DEBUG_LOGGING:
                            self.get_logger().info(f"Znaleziono frontier w pobliżu celu (odległość: {distance:.1f} komórek)")
                        return True
            
            # Cel nie istnieje w żadnym klastrze
            if self.DEBUG_LOGGING:
                self.get_logger().info(f"Cel {self._current_target} nie znaleziony w {len(clusters)} klastrach")
            return False
        except Exception as e:
            self.get_logger().error(f"Błąd sprawdzania istnienia celu: {e}")
            return False
    
    def _handle_navigation_result(self, result) -> None:
        """
        Obsługuje wynik nawigacji z obsługą lokalnej eksploracji.
        Rozróżnia między sukcesem/niepowodzeniem głównej eksploracji a lokalnej eksploracji.
        """
        try:
            
            if self.DEBUG_LOGGING:
                self.get_logger().info(f"Obsługuje wynik nawigacji: {result}")
            
            result = self.nav.getResult()
            time.sleep(0.2)  # Krótkie opóźnienie dla stabilności
            success = (result == TaskResult.SUCCEEDED)
            aborted = (result == TaskResult.FAILED)
            
            if success:
                self.exploration_stats['successful_navigations'] += 1

                # BLOK: Sukces głównej eksploracji
                self.exploration_stats['frontiers_visited'] += 1
                self.get_logger().info(f"Sukces! Cel osiągnięty: {self._current_target}")
                
                # Dodaj poprzedni cel do blacklisty po udanej nawigacji
                if self._current_target:
                    self.get_logger().info(f"🚫 Blacklistuję osiągnięty cel: {self._current_target}")
                    self._blacklist_neighbors(self._current_target, radius_m=self.BLACKLIST_RADIUS)
                    self.get_logger().info(f"🚫 Rozmiar blacklisty po dodaniu: {len(self._frontier_blacklist)}")
                
                # Dodaj aktualną pozycję do listy odwiedzonych miejsc
                current_pos = self._get_robot_pose()
                if current_pos:
                    self._visited_positions.append(current_pos)
                    # Ogranicz listę do ostatnich 10 pozycji
                    if len(self._visited_positions) > 10:
                        self._visited_positions = self._visited_positions[-10:]
                    
                    self.get_logger().info(f"Dodano pozycję {current_pos} do listy odwiedzonych ({len(self._visited_positions)} pozycji)")
                
                # Reset stanu
                if self._current_target in self._frontier_attempt_history:
                    del self._frontier_attempt_history[self._current_target]
                self._failed_frontier_attempts = 0
                
                # NOWY BLOK: Aktualizacja indeksu optymalnej trasy
                if self._cluster_route_optimized:
                    self._current_cluster_index += 1
                    self.get_logger().info(f"Przechodzę do następnego klastra w optymalnej trasie: {self._current_cluster_index}")
                
                # USUNIĘTY BLOK: Blacklistowanie pozycji robota (teraz wykonywane ciągle)
                # Ten blok został przeniesiony do _track_position()
                
                # Reset stanu nawigacji
                self._current_target = None
                self._current_target_score = -math.inf

                self._start_next_navigation()
                        
            elif aborted or result == TaskResult.FAILED:
                # BLOK: Niepowodzenie nawigacji
                self.exploration_stats['failed_navigations'] += 1
                self.get_logger().warning(f"Niepowodzenie nawigacji do {self._current_target}! Status: {result}")
                
                
                # BLOK: Zarządzanie powtórnymi próbami
                if self._current_target:
                    if self._current_target not in self._frontier_attempt_history:
                        self._frontier_attempt_history[self._current_target] = 0
                    self._frontier_attempt_history[self._current_target] += 1
                    
                    attempts = self._frontier_attempt_history[self._current_target]
                    
                    if attempts < self._max_attempts_per_frontier:
                        # Powtórz próbę
                        _, clusters, _, is_fresh = self._get_cached_frontiers()
                        if is_fresh and self._is_target_still_valid(clusters):
                            map_params = self._get_map_params()
                            if map_params and len(map_params) == 3:
                                goal = self._cell_to_world(self._current_target, *map_params)
                                if self._path_is_feasible(goal):
                                    current_target_backup = self._current_target
                                    current_score_backup = self._current_target_score
                                    
                                    self._current_target = None
                                    self._current_target_score = -math.inf
                                    
                                    self._send_goal(*goal, cell=current_target_backup, score=current_score_backup, retry_attempt=True)
                                    return
                    else:
                        # Przekroczono limit prób
                        self._blacklist_neighbors(self._current_target, radius_m=0.2)
                
                # Reset stanu i znajdź nowy cel
                self._current_target = None
                self._current_target_score = -math.inf
                self._start_next_navigation()
        except Exception as e:
            self.get_logger().error(f"Błąd obsługi wyniku nawigacji: {e}")
            # Reset stanu nawigacji przy błędzie
            self._current_target = None
            self._current_target_score = -math.inf
    
    def _calculate_frontier_score(self, cell: Tuple[int,int]) -> float:
        """
        Oblicza score dla danego frontiera na podstawie information gain i odległości.
        
        Args:
            cell: Komórka frontiera (r, c)
            frontiers: Lista wszystkich frontierów
            map_params: Parametry mapy (origin_x, origin_y, resolution)
            
        Returns:
            Obliczony score frontiera
        """
        def gauss_cap(distance, d_top=5.0, d_base=8.0, sigma=0.8):
            # Obsługa zarówno float, jak i array
            R = np.array(distance, ndmin=1)
            r_top = d_top / 2
            r_base = d_base / 2

            Z = np.zeros_like(R, dtype=float)

            # Płaska góra
            Z[R <= r_top] = 1.0

            # Wygładzona część między r_top a r_base
            mask = (R > r_top) & (R <= r_base)
            Z[mask] = np.exp(-((R[mask] - r_top) ** 2) / (2 * sigma ** 2))

            # Poza bazą: Z = 0
            # Jeśli wejście było skalarem, zwróć skalar
            return Z[0] if np.isscalar(distance) else Z
        
        try:
            robot_pose = self._get_robot_pose()
            if not robot_pose:
                return 0.0
                
            _,_, map_params, _ = self._get_cached_frontiers()
            ox, oy, res = map_params
            rx, ry = robot_pose
            cx, cy = self._cell_to_world(cell, ox, oy, res)
            
            distance = math.sqrt((cx - rx)**2 + (cy - ry)**2)
            info_gain = self._frontier_cache.get_num_of_elements_in_clusters(cell)
            self.score_dist = (self.DIST_WEIGHT * gauss_cap(distance, d_top=2.0, d_base=10.0, sigma=1.5)) if info_gain > 20 else 0 # "Gauss Cap" - spłaszczona funkcja rozkładu normalnego z ograniczeniami
            
            score = (self.INFO_WEIGHT * info_gain + self.score_dist) #nieliniowe 
        
            return float(score)
            
        except Exception as e:
            if self.DEBUG_LOGGING:
                self.get_logger().error(f"Błąd obliczania score frontiera: {e}")
            return 0.0
    
    def _world_to_cell(self, world_pos: Tuple[float, float], origin_x: float, origin_y: float, resolution: float) -> Tuple[int, int]:
        """
        Konwertuje współrzędne świata na komórkę mapy.
        
        Args:
            world_pos: Pozycja w świecie (x, y)
            origin_x, origin_y: Punkt początkowy mapy
            resolution: Rozdzielczość mapy (m/piksel)
            
        Returns:
            Komórka mapy (r, c)
        """
        x, y = world_pos
        c = int((x - origin_x) / resolution)
        r = int((y - origin_y) / resolution)
        return r, c

    def _optimize_exploration_order(self, clusters: List[List[Tuple[int,int]]]) -> List[Tuple[int,int]]:
        """
        Optymalizuje kolejność odwiedzania klastrów używając prostego algorytmu najbliższego sąsiada.
        """
        if not clusters:
            return []
        
        robot_pos = self._get_robot_pose()
        if not robot_pos:
            return [cluster[0] for cluster in clusters]
        
        map_params = self._get_map_params()
        if not map_params:
            return [cluster[0] for cluster in clusters]
        
        ox, oy, res = map_params
        
        # Konwertuj centroidy klastrów na współrzędne świata
        cluster_positions = []
        for cluster in clusters:
            if cluster:
                centroid = cluster[0]
                world_pos = self._cell_to_world(centroid, ox, oy, res)
                cluster_positions.append((centroid, world_pos))
        
        if not cluster_positions:
            return []
        
        # Algorytm najbliższego sąsiada
        ordered_clusters = []
        current_pos = robot_pos
        remaining = cluster_positions.copy()
        
        while remaining:
            # Znajdź najbliższy klaster
            closest_idx = 0
            min_distance = math.hypot(
                current_pos[0] - remaining[0][1][0],
                current_pos[1] - remaining[0][1][1]
            )
            
            for i, (_, world_pos) in enumerate(remaining[1:], 1):
                distance = math.hypot(
                    current_pos[0] - world_pos[0],
                    current_pos[1] - world_pos[1]
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
            
            # Dodaj do uporządkowanej listy
            closest_cluster = remaining.pop(closest_idx)
            ordered_clusters.append(closest_cluster[0])  # centroid
            current_pos = closest_cluster[1]  # world position
        
        self.get_logger().info(f"📋 Zoptymalizowano kolejność {len(ordered_clusters)} klastrów")
        return ordered_clusters

    def _get_exploration_zone(self, robot_pos: Tuple[float, float], radius: float = 4.0) -> List[Tuple[int,int]]:
        """
        Zwraca frontiers w określonej strefie wokół robota.
        """
        frontiers, clusters, map_params, is_fresh = self._get_cached_frontiers()
        if not is_fresh or not clusters:
            return []
        
        ox, oy, res = map_params
        zone_clusters = []
        
        for cluster in clusters:
            if cluster:
                centroid = cluster[0]
                cx, cy = self._cell_to_world(centroid, ox, oy, res)
                distance = math.hypot(cx - robot_pos[0], cy - robot_pos[1])
                
                if distance <= radius:
                    zone_clusters.append(centroid)
        
        return zone_clusters
    
    
    def _find_best_frontier(self) -> Tuple[Optional[Tuple[int,int]], float]:
        """
        Znajduje najlepszy frontier z optymalizacją trasy klastrów po odkryciu nowych klastrów.
        """
        try:
            self.get_logger().info("_find_best_frontier() START")
            
            if not self._map:
                self.get_logger().warning("Brak mapy w _find_best_frontier")
                return None, -math.inf
                
            pose = self._get_robot_pose()
            if not pose:
                self.get_logger().warning("Brak pozycji robota w _find_best_frontier")
                return None, -math.inf
            
            rx, ry = pose
            self.get_logger().info(f"Pozycja robota: ({rx:.2f}, {ry:.2f})")
        
            # Użyj cache'u frontierów
            frontiers, clusters, map_params, is_fresh = self._get_cached_frontiers()
            
            self.get_logger().info(f"Cache: fresh={is_fresh}, frontiers={len(frontiers)}, clusters={len(clusters)}")
            
            if not is_fresh or not clusters:
                self.get_logger().warning("Cache nieświeży lub brak klastrów")
                return None, -math.inf
        
            ox, oy, res = map_params
            self.get_logger().info(f"Parametry mapy: origin=({ox:.2f}, {oy:.2f}), res={res:.4f}")
        
            # Filtruj według blacklisty
            available_clusters = [c for c in clusters if c[0] not in self._frontier_blacklist]
            
            current_cluster_count = len(available_clusters)
            if not hasattr(self, '_last_cluster_count'):
                self._last_cluster_count = current_cluster_count
            
            self.get_logger().info(f"Dostępne klastry (po filtrowaniu blackliście): {len(available_clusters)}/{len(clusters)}")
            self.get_logger().info(f"Rozmiar blackliście: {len(self._frontier_blacklist)}")
            
            if not available_clusters:
                self.get_logger().warning("Brak dostępnych klastrów po filtrowaniu blackliście")
                return None, -math.inf
        
            # Sprawdź lokalne frontiers NAJPIERW
            if pose:
                local_frontiers = self._get_exploration_zone(pose, radius=3.0)
                if local_frontiers:
                    for frontier in local_frontiers:
                        if frontier not in self._frontier_blacklist:
                            score = self._calculate_frontier_score(frontier)
                            if score > self.SCORE_THRESHOLD * 0.7:  # Niższy próg dla lokalnych
                                self.get_logger().info(f"🎯 Wybrano lokalny frontier: {frontier}")
                                return frontier, score
            
            # Optymalizacja trasy TYLKO RAZ, na początku
            optimal_order = None
            if len(available_clusters) > 3:
                optimal_order = self._optimize_exploration_order(available_clusters)
                if optimal_order:
                    # Sprawdź pierwszy z optymalnej kolejności
                    first_optimal = optimal_order[0]
                    first_score = self._calculate_frontier_score(first_optimal)
                    if first_score > self.SCORE_THRESHOLD * 0.8:  # 80% progu
                        self.get_logger().info(f"🎯 Wybieram pierwszy z optymalnej kolejności: {first_optimal}")
                        return first_optimal, first_score
            
            # Dynamiczne progi odległości
            exploration_time = self.get_current_time() - self.exploring_time_start
            if exploration_time < 30:
                min_distance_to_previous = 1.0
                min_distance_to_robot = 0.5
                min_distance_to_visited = 1.5  # NAPRAWIONE: Dodane dla wczesnej fazy
            else:
                min_distance_to_previous = 3.0
                min_distance_to_robot = 2.5
                min_distance_to_visited = 2.5
            
            # Oceń dostępne klastry
            best_cell = None
            best_score = -math.inf
            scores_debug = []
            
            self.get_logger().info(f"Rozpoczynam ocenę {len(available_clusters)} klastrów (progi: prev={min_distance_to_previous}m, robot={min_distance_to_robot}m, visited={min_distance_to_visited}m)...")
            
            # NAPRAWIONE: Użyj optymalnej kolejności jeśli istnieje
            clusters_to_evaluate = optimal_order if optimal_order else [cluster[0] for cluster in available_clusters]
            
            for i, centroid_cell in enumerate(clusters_to_evaluate):
                # NAPRAWIONE: Jedna konwersja na początku
                cx, cy = self._cell_to_world(centroid_cell, ox, oy, res)
                self.get_logger().info(f"Ocena klastra {i+1}/{len(clusters_to_evaluate)}: {centroid_cell} ({cx:.2f}, {cy:.2f})")
                
                # Sprawdź odległość od robota
                distance_to_robot = math.sqrt((cx - rx)**2 + (cy - ry)**2)
                if distance_to_robot < min_distance_to_robot:
                    self.get_logger().info(f"Klaster {centroid_cell} za blisko robota ({distance_to_robot:.2f}m < {min_distance_to_robot}m) - pomijam")
                    continue
                
                # Sprawdź odległość od poprzedniego celu
                if hasattr(self, '_previous_target') and self._previous_target is not None:
                    prev_x, prev_y = self._cell_to_world(self._previous_target, ox, oy, res)
                    distance_to_previous = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    if distance_to_previous < min_distance_to_previous:
                        self.get_logger().info(f"🚫 Klaster {centroid_cell} za blisko poprzedniego celu {self._previous_target} ({distance_to_previous:.2f}m < {min_distance_to_previous}m) - pomijam")
                        continue
                
                # Sprawdź odległość od odwiedzonych pozycji
                too_close_to_visited = False
                if hasattr(self, '_visited_positions') and self._visited_positions:
                    for visited_pos in self._visited_positions:
                        distance_to_visited = math.sqrt((cx - visited_pos[0])**2 + (cy - visited_pos[1])**2)
                        if distance_to_visited < min_distance_to_visited:
                            self.get_logger().info(f"🚫 Klaster {centroid_cell} za blisko odwiedzonej pozycji ({distance_to_visited:.1f}m)")
                            too_close_to_visited = True
                            break
                
                if too_close_to_visited:
                    continue
                
                # Oblicz score
                info_gain = self._frontier_cache.get_num_of_elements_in_clusters(centroid_cell)
                score = self._calculate_frontier_score(centroid_cell)
        
                scores_debug.append({
                    'cell': centroid_cell,
                    'info_gain': info_gain,
                    'path_length': distance_to_robot,
                    'score_dist': getattr(self, 'score_dist', 0),  # NAPRAWIONE: Bezpieczny dostęp
                    'total_score': score,
                    'position': (cx, cy)
                })
                
                if score > best_score:
                    best_score = score
                    best_cell = centroid_cell
                    self.get_logger().info(f"Nowy najlepszy: {centroid_cell} ze score {score:.2f}")
        
            self.get_logger().info(f"Najlepszy frontier: {best_cell}, score: {best_score:.2f}")
        
            # Wizualizuj informacje o frontierach
            if self.ENABLE_VISUALIZATION and scores_debug:
                self.utils.visualize_frontier_info(scores_debug, best_cell, self.info_pub, None)
        
            return best_cell, best_score
            
        except Exception as e:
            self.get_logger().error(f"Błąd w _find_best_frontier: {e}")
            return None, -math.inf
    
    def _explore(self) -> None:
        """
        Unified exploration and continuous evaluation method.
        Handles both the main exploration loop and continuous target evaluation.
        """
        try:
            current_time = self.get_current_time()
            
            # Check if map is available
            if not self._map:
                self.get_logger().warning("Brak mapy w _explore()")
                return
                
            # CASE 1: Active navigation target exists
            if self._current_target is not None:
                self.get_logger().info(f"🧭🚀 _explore() START [{current_time:.1f}] - cel: {self._current_target}")
                
                # Validate if current target still exists in frontier database
                _, clusters, _, is_fresh = self._get_cached_frontiers()
                if not is_fresh:
                    self.get_logger().warning("Cache frontierów nieświeży w _explore()")
                    return
                    
                if self._current_target and not self._is_target_cell_valid(self._current_target, clusters):
                    self.get_logger().info(f"🧭🚀 Cel {self._current_target} nie istnieje w bazie danych, wybieram nowy")
                    self._current_target = None
                    self._current_target_score = -math.inf
                    self._start_next_navigation()
                    return
                
                # Check if navigation is complete
                nav_complete = self.nav.isTaskComplete()
                if nav_complete:
                    result = self.nav.getResult()
                    self.get_logger().info(f"🧭🚀 Nawigacja zakończona: {result}")
                    self._handle_navigation_result(result)
                    return
                
                # CONTINUOUS EVALUATION: Only if enough time has passed
                time_since_last_eval = current_time - getattr(self, '_last_continuous_eval', 0.0)
                if time_since_last_eval >= 2.0:  # Minimum 2s between evaluations
                    self._last_continuous_eval = current_time
                    
                    if self.DEBUG_LOGGING:
                        self.get_logger().info(f"🔎 Evaluating current navigation [{current_time:.1f}]")
                    
                    # Pre-calculate scores for all clusters
                    for cluster in clusters:
                        if cluster:  # Make sure cluster is not empty
                            self._calculate_frontier_score(cluster[0])
                    
                    # Check if current target is still valid
                    if not self._is_target_still_valid(clusters):
                        self.get_logger().info(f"ℹ️ Aktualny cel {self._current_target} nie jest już aktualny")
                        
                        # Calculate new best frontier
                        new_best_cell, new_best_score = self._find_best_frontier()
                        if new_best_cell is not None and new_best_cell != self._current_target:
                            self.get_logger().info(
                                f"🔄 Zmieniam cel: {self._current_target} → {new_best_cell} "
                                f"(score: {self._current_target_score:.2f} → {new_best_score:.2f})"
                            )
                            # Reset state before sending new goal
                            self._current_target = None
                            self._current_target_score = -math.inf
                            map_params = self._get_map_params()
                            if map_params and len(map_params) == 3:
                                goal = self._cell_to_world(new_best_cell, *map_params)
                                self._send_goal(*goal, cell=new_best_cell, score=new_best_score)
                            return
                        else:
                            # If no better frontier found, cancel current navigation and restart
                            if not self.nav.isTaskComplete():
                                self.nav.cancelTask()
                            self._start_next_navigation()
                            return
                    
                    # Check if there's a better frontier
                    pose = self._get_robot_pose()
                    if pose:
                        best_cell, best_score = self._find_best_frontier()
                        
                        # Skip if new frontier is very close to current target
                        if self._current_target and best_cell:
                            if abs(best_cell[0] - self._current_target[0]) <= 1 and abs(best_cell[1] - self._current_target[1]) <= 1:
                                if self.DEBUG_LOGGING:
                                    self.get_logger().info("ℹ️ Nowy frontier w pobliżu aktualnego celu - kontynuuję")
                                return
                        
                        # Handle different frontier comparison cases
                        if best_cell == self._current_target:
                            if self.DEBUG_LOGGING:
                                self.get_logger().info("ℹ️ Aktualny cel nadal najlepszy")
                            return
                        elif best_cell is None:
                            if self.DEBUG_LOGGING:
                                self.get_logger().info("ℹ️ Brak nowych frontierów - kontynuuję")
                            return
                        else:
                            # Check if new goal is significantly better
                            improvement = best_score - self._current_target_score
                            
                            if improvement > self.MIN_SCORE_IMPROVEMENT:
                                self.get_logger().info(
                                    f"🚀 Lepszy frontier: {self._current_target}→{best_cell} "
                                    f"(score: {self._current_target_score:.2f}→{best_score:.2f}, +{improvement:.2f})"
                                )
                                
                                map_params = self._get_map_params()
                                if map_params and len(map_params) == 3:
                                    goal = self._cell_to_world(best_cell, *map_params)
                                    self._send_goal(*goal, cell=best_cell, score=best_score)
                            else:
                                if self.DEBUG_LOGGING:
                                    self.get_logger().info(f"ℹ️ Niewystarczająca poprawa: +{improvement:.2f}")
                
                # Throttling check - don't attempt navigation too frequently
                time_since_last_attempt = current_time - getattr(self, '_last_navigation_attempt', 0.0)
                if time_since_last_attempt < 1.0:
                    if self.DEBUG_LOGGING:
                        self.get_logger().info(f"🧭🚀 Throttling navigation ({time_since_last_attempt:.1f}s)")
                    return
                
            # CASE 2: No active target, start new navigation
            else:
                self.get_logger().info("🧭🚀 Brak aktywnego celu, rozpoczynam nową nawigację")
                self._last_navigation_attempt = current_time
                self.retry_attempt = True  # Reset retry attempt flag
                self._start_next_navigation()
                
        except Exception as e:
            self.get_logger().error(f"❌ Błąd w _explore(): {e}")
            # Reset navigation state on error
            self._current_target = None
            self._current_target_score = -math.inf
    
    def _map_to_numpy(self, occupancy_grid: OccupancyGrid) -> Tuple[np.ndarray, float, float, float]:
        """
        Konwertuje OccupancyGrid ROS2 do numpy array.
        
        Args:
            occupancy_grid: Mapa zajętości z ROS2
            
        Returns:
            Tuple zawierająca (grid, origin_x, origin_y, resolution)
        """
        data = np.array(occupancy_grid.data, dtype=np.int8)
        grid = data.reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        
        resolution = occupancy_grid.info.resolution
        origin_x = occupancy_grid.info.origin.position.x
        origin_y = occupancy_grid.info.origin.position.y
        
        return grid, origin_x, origin_y, resolution
    
    def _cell_to_world(self, cell: Tuple[int,int], origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
        """
        Konwertuje komórkę mapy na współrzędne świata.
        
        Args:
            cell: Komórka mapy (r, c)
            origin_x, origin_y: Punkt początkowy mapy
            resolution: Rozdzielczość mapy (m/piksel)
            
        Returns:
            Współrzędne świata (x, y)
        """
        r, c = cell
        x = origin_x + (c + 0.5) * resolution
        y = origin_y + (r + 0.5) * resolution
        return x, y
    
    def _get_map_params(self) -> Tuple[float, float, float]:
        """
        Zwraca parametry aktualnej mapy z obsługą błędów.
        
        Returns:
            Tuple zawierająca (origin_x, origin_y, resolution) lub wartości domyślne
        """
        if not self._map:
            self.get_logger().warning("_get_map_params(): Brak mapy!")
            return 0.0, 0.0, 0.05
        
        try:
            _, ox, oy, res = self._map_to_numpy(self._map)
            return ox, oy, res
        except Exception as e:
            self.get_logger().error(f"Błąd w _get_map_params(): {e}")
            return 0.0, 0.0, 0.05
    
    def _path_is_feasible(self, goal: Tuple[float, float]) -> bool:
        """
        Sprawdza czy ścieżka do celu jest wykonalna używając planera Nav2.
        
        Args:
            goal: Współrzędne celu (x, y)
            
        Returns:
            True jeśli ścieżka istnieje, False w przeciwnym razie
        """
        try:
            pose = self._get_robot_pose_full()
            if pose is not None:
                x, y, yaw = pose
                
                # BLOK: Przygotowanie pozycji startowej
                init_pose = PoseStamped()
                init_pose.header.frame_id = 'map'
                init_pose.header.stamp = self.nav.get_clock().now().to_msg()
                init_pose.pose.position.x = x
                init_pose.pose.position.y = y
                init_pose.pose.position.z = 0.0
                init_pose.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
                )
    
                # BLOK: Przygotowanie pozycji docelowej
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
                goal_pose.pose.position.x = goal[0]
                goal_pose.pose.position.y = goal[1]
                goal_pose.pose.position.z = 0.0
                goal_pose.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
                )
    
                # BLOK: Sprawdzenie wykonalności ścieżki
                plan = self.nav.getPath(init_pose, goal_pose)
                if plan and len(plan.poses) > 0:
                    self.get_logger().info(f"Ścieżka do celu ({goal[0]:.2f}, {goal[1]:.2f}) jest wykonalna")
                    return True
                else:
                    self.get_logger().info(f"Brak ścieżki do celu ({goal[0]:.2f}, {goal[1]:.2f})")
                    return False
            else:
                self.get_logger().warning("_path_is_feasible(): Brak pozycji robota!")
                return False
                
        except Exception as e:
            self.get_logger().warning(f"Błąd sprawdzania ścieżki: {e}")
            return False
    
    def _blacklist_area(self, cell: Tuple[int,int], radius_m: float) -> None:
        """
        Blacklistuje obszar wokół komórki w podanym promieniu.
        
        Args:
            cell: Komórka centralna (r, c)
            radius_m: Promień blacklistowania w metrach
        """
        if not self._map:
            return
            
        res = self._map.info.resolution
        max_r, max_c = self._map.info.height, self._map.info.width
        radius_cells = int(radius_m / res)
        r0, c0 = cell
        
        cells_blacklisted = 0
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                distance = math.sqrt(dr*dr + dc*dc)
                if distance <= radius_cells:
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < max_r and 0 <= cc < max_c:
                        self._frontier_blacklist.add((rr, cc))
                        cells_blacklisted += 1
        
        if self.DEBUG_LOGGING:
            self.get_logger().info(f"Blacklistuję {cells_blacklisted} komórek wokół {cell} (promień: {radius_m}m)")
    
    def _blacklist_neighbors(self, cell: Tuple[int,int], radius_m: float = None) -> None: #type: ignore
        """
        Blacklistuje sąsiedztwo komórki z domyślnym promieniem.
        
        Args:
            cell: Komórka do blacklistowania (r, c)
            radius_m: Promień blacklistowania (domyślnie BLACKLIST_RADIUS)
        """
        if radius_m is None:
            radius_m = self.BLACKLIST_RADIUS
        self._blacklist_area(cell, radius_m)
        
        if self.DEBUG_LOGGING:
            self.get_logger().info(f"Blacklistuję sąsiedztwo {cell} w promieniu {radius_m}m")

    def _start_next_navigation(self) -> None:
        """
        Rozpoczyna nawigację do kolejnego frontiera z kompleksową walidacją.
        Implementuje logikę pierwszego ruchu, sprawdza granice mapy i wykonalność ścieżki.
        """
        current_time = self.get_current_time()
        self.get_logger().info(f"_start_next_navigation() START [{current_time:.1f}]")
        
    
        if not self._map:
            self.get_logger().warning("Brak mapy w _start_next_navigation")
            return
        frontiers, clusters, map_params, is_fresh = self._get_cached_frontiers()
        # BLOK: Sprawdzenie czy już mamy aktywny cel
        if self._current_target is not None:
            self.get_logger().info(f"Już mamy aktywny cel {self._current_target} - sprawdzam status")
            result = self.nav.isTaskComplete()
            time.sleep(0.2)  # Krótkie opóźnienie dla stabilności
            if not result:
                self.get_logger().info(f"Nawigacja do {self._current_target} w toku - czekam")
                return
            else:
                # Nawigacja zakończona - obsłuż wynik
                result = self.nav.getResult()
                time.sleep(0.2)  # Krótkie opóźnienie dla stabilności
                self.get_logger().info(f"Wykryto zakończoną nawigację w _start_next_navigation: {result}")
                self.get_logger().info("LINIA 3090 - obsługuje wynik nawigacji =======================================")
                self._handle_navigation_result(result)
                return
                
        # BLOK: Wyszukiwanie najlepszego frontiera
        self.get_logger().info("Rozpoczynam wyszukiwanie najlepszego frontiera...")
        best_cell, best_score = self._find_best_frontier()
        
        self.get_logger().info(f"Najlepszy frontier: {best_cell}, score: {best_score:.2f}, próg: {self.SCORE_THRESHOLD}")
        
        # BLOK: Diagnoza braku frontiera
        if best_cell is None:
            self.get_logger().warning("Brak najlepszego frontiera - sprawdzam przyczyny...")
            
            frontiers, clusters, _, is_fresh = self._get_cached_frontiers()
            self.get_logger().info(f"Cache: fresh={is_fresh}, frontiers={len(frontiers)}, clusters={len(clusters)}")
            
            if not is_fresh:
                self.get_logger().warning("Cache nieświeży - czekam na nowe dane")
                return
            
            if not clusters:
                self.get_logger().warning("Brak klastrów frontierów")
                return
            
            available_clusters = [c for c in clusters if c[0] not in self._frontier_blacklist]
            self.get_logger().info(f"Dostępne klastry (po blackliście): {len(available_clusters)}/{len(clusters)}")
            
            if not available_clusters:
                self.get_logger().warning("Wszystkie klastry na blackliście!")
                self._handle_no_suitable_frontier()
                return
        
        # BLOK: Logika pierwszego ruchu - na początku akceptuj dowolny frontier
        if best_cell and (best_score >= self.SCORE_THRESHOLD or self.get_current_time()-self.exploring_time_start < 30):
            self.get_logger().info(f"Znaleziono frontier z wymaganym score: {best_cell}, score: {best_score:.2f}")
            
            self._failed_frontier_attempts = 0
            
            map_params = self._get_map_params()
            if not map_params or len(map_params) != 3:
                self.get_logger().error("Brak parametrów mapy")
                return
            
            goal = self._cell_to_world(best_cell, *map_params)
            self.get_logger().info(f"Cel w świecie: {goal}")
            
            # BLOK: Walidacja granic mapy
            if not self._is_goal_within_map_bounds(*goal):
                self.get_logger().warning(f"Cel {goal} poza granicami mapy - blacklistuję {best_cell}")
                self._blacklist_neighbors(best_cell)
                self._recursive_navigation_attempt()
                return
            
            # BLOK: Walidacja wykonalności ścieżki
            self.get_logger().info(f"Sprawdzam wykonalność ścieżki do {goal}")
            if not self._path_is_feasible(goal):
                self.get_logger().warning(f"Ścieżka do {goal} nieosiągalna - blacklistuję {best_cell}")
                self._blacklist_neighbors(best_cell)
                self._recursive_navigation_attempt()
                return
            self._send_goal(*goal, cell=best_cell, score=best_score, retry_attempt=self.retry_attempt)
            return
        
        # BLOK: Brak odpowiedniego frontiera
        self.get_logger().warning(f"Brak frontiera z wymaganym score (najlepszy: {best_score:.2f} < {self.SCORE_THRESHOLD})")
        self._handle_no_suitable_frontier()
    
    def _recursive_navigation_attempt(self) -> None:
        """
        Bezpieczne ponowienie próby nawigacji z ochroną przed nieskończoną rekursją.
        Ogranicza głębokość rekursji do maksymalnie 2 poziomów.
        """
        if not hasattr(self, '_recursion_depth'):
            self._recursion_depth = 0
        
        if self._recursion_depth < 2:  # Maksymalnie 2 poziomy
            self._recursion_depth += 1
            self._start_next_navigation()
            self._recursion_depth -= 1
        else:
            self.get_logger().warning("Zatrzymano rekursję w nawigacji")
            self._recursion_depth = 0
    
    def _handle_no_suitable_frontier(self) -> None:
        """
        Obsługuje sytuację braku odpowiedniego frontiera.
        Implementuje adaptacyjne zwiększanie wag i finalną decyzję o zakończeniu eksploracji.
        """
        frontiers, _, _, is_fresh = self._get_cached_frontiers()
        
        if is_fresh and frontiers:
            # BLOK: Są frontiere, ale żaden nie ma wymaganego score
            self.get_logger().info(f"Frontiers: {len(frontiers)}, score poniżej progu")
            
            if self.attempt_blacklist < 1:
                self.attempt_blacklist += 1
                # Opcjonalne czyszczenie blackliście (zakomentowane)
            
            # BLOK: Adaptacyjne zwiększanie wagi information gain
            if len(frontiers) > 50:
                self.INFO_WEIGHT += self.ADAPTIVE_INFO_GAIN
                self.get_logger().info(f"Zwiększono INFO_WEIGHT do {self.INFO_WEIGHT}")
            
            self._failed_frontier_attempts += 1
            self.get_logger().info(
                f"Brak frontierów z score>={self.SCORE_THRESHOLD} "
                f"(próba {self._failed_frontier_attempts}/{self._max_failed_attempts})"
            )
            
            # BLOK: Sprawdzenie limitu nieudanych prób
            if self._failed_frontier_attempts >= self._max_failed_attempts:
                self.get_logger().info("Limit nieudanych prób osiągnięty - kończę eksplorację")
                self._finish_exploration_and_save_map()
            
        elif is_fresh and not frontiers:
            # BLOK: Brak frontierów w ogóle
            self.get_logger().info("Brak frontierów - eksploracja zakończona")
            self._finish_exploration_and_save_map()
        else:
            # BLOK: Cache nieświeży
            self.get_logger().info("Czekam na świeże dane frontierów...")
        
    def _finish_exploration_and_save_map(self) -> None:
        """
        Kończy eksplorację i zapisuje wszystkie wyniki.
        Wywołuje serwisy SLAM do serializacji mapy, zatrzymuje timery i czyści zasoby.
        """
        if self.ENABLE_VISUALIZATION:
            self._clear_all_markers()
            
        # BLOK: Przygotowanie nazwy pliku wykresu
        graph_filename = os.path.join(
            self.MAP_SAVE_DIR, 
            f"exploration_graph_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )

        if not self._map_saved:
            # NOWY BLOK: Logowanie statystyk blackliście pozycji
            if hasattr(self, '_total_distance_traveled'):
                robot_blacklist_count = 0
                if self._map:
                    try:
                        map_params = self._get_map_params()
                        if map_params and len(map_params) == 3:
                            # Policz ile pozycji robota jest na blackliście
                            for pos_data in getattr(self, '_robot_position_history', []):
                                if isinstance(pos_data, dict) and 'x' in pos_data and 'y' in pos_data:
                                    pos = (pos_data['x'], pos_data['y'])
                                    robot_cell = self._world_to_cell(pos, *map_params)
                                    if robot_cell in self._frontier_blacklist:
                                        robot_blacklist_count += 1
                    except Exception as e:
                        self.get_logger().error(f"Błąd liczenia blackliście pozycji: {e}")
                
                self.get_logger().info(
                    f"📊 Statystyki pozycji robota: przejechano {self._total_distance_traveled:.1f}m, "
                    f"blacklistowano {robot_blacklist_count} pozycji"
                )
            
            # BLOK: Zapisywanie mapy przez serwisy SLAM
            if self._map and self.MAP_SAVE_ENABLED:
                self._map_saved = True
                self._serialize_slam_map()
                self.get_logger().info("Mapa zapisana pomyślnie")
    
            # BLOK: Zapisywanie podsumowania eksploracji
            self.utils.save_exploration_summary(self.exploration_stats, self._map_to_numpy)
            self.exploration_stats['frontiers_not_visited'] = len(self._frontier_cache._frontiers)
    
            # BLOK: Zapisywanie wizualizacji trasy
            if self._map is not None:
                self.utils.save_path_visualization(self.exploration_stats, self._map, self._map_to_numpy,  self._frontier_cache._frontiers)
            else:
                self.get_logger().warning("Nie można zapisać wizualizacji trasy - brak mapy")
    
            # BLOK: Zapisywanie wykresu
            self.graph.save_plot(graph_filename)
            self.graph.close()
            
            # BLOK: Anulowanie aktywnych nawigacji
            try:
                if not self.nav.isTaskComplete():
                    self.nav.cancelTask()
                    self.get_logger().info("Anulowano aktywną nawigację")
                    self.nav.destroyNode()
                    
                    # Zatrzymanie robota
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info("Zatrzymano robota")
                    
                    # Obliczenie czasu eksploracji
                    self.exploring_time = self.nav.get_clock().now().seconds_nanoseconds()[0] - self.exploring_time_start
                    self.get_logger().info(f"Czas eksploracji: {self.exploring_time} sekund")
            except Exception as e:
                self.get_logger().warning(f"Błąd anulowania nawigacji: {e}")
                 # BLOK: Zatrzymywanie wszystkich timerów
        if hasattr(self, '_timer') and self._timer:
            self._timer.cancel()
            self.get_logger().info("Zatrzymano główny timer eksploracji")
            
        if hasattr(self, '_viz_timer') and self._viz_timer:
            self._viz_timer.cancel()
            self.get_logger().info("Zatrzymano timer wizualizacji")
            
        if hasattr(self, 'position_timer') and self.position_timer:
            self.position_timer.cancel()
            self.get_logger().info("Zatrzymano timer trackingu pozycji")
            
            self.get_logger().info("Eksploracja zakończona!")
            self.destroy_node()
                    
        # Reset licznika dla ewentualnej kolejnej eksploracji
        self._failed_frontier_attempts = 0
    
    def _serialize_slam_map(self) -> None:
        """
        Wywołuje serwisy slam_toolbox do serializacji i zapisu mapy.
        Przygotowuje żądania asynchroniczne z callbackami obsługi odpowiedzi.
        """
        try:
            # BLOK: Tworzenie klientów serwisu
            serialize_client = self.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
            save_map_client = self.create_client(SaveMap, '/slam_toolbox/save_map')
            
            # BLOK: Sprawdzenie dostępności serwisów
            if not serialize_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warning("Serwis /slam_toolbox/serialize_map niedostępny")
                return
            
            if not save_map_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warning("Serwis /slam_toolbox/save_map niedostępny")
                return
            
            # BLOK: Przygotowanie żądania serializacji
            serialize_request = SerializePoseGraph.Request()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            serialize_request.filename = os.path.join(self.MAP_SAVE_DIR,f"{timestamp}_serialized")
            
            self.get_logger().info(f"Wysyłam żądanie serializacji mapy: {serialize_request.filename}")
            
            # Wywołanie serwisu asynchronicznie
            future = serialize_client.call_async(serialize_request)
            future.add_done_callback(self._handle_serialize_response)
            
            # BLOK: Przygotowanie żądania zapisu mapy
            save_request = SaveMap.Request()
            save_request.name = String()
            save_request.name.data = os.path.join(self.MAP_SAVE_DIR, f"{timestamp}_map")
            self.get_logger().info(f"Wysyłam żądanie zapisu mapy: {save_request.name}")
            
            future_save = save_map_client.call_async(save_request)
            future_save.add_done_callback(self._handle_save_response)
            
        except Exception as e:
            self.get_logger().error(f"Błąd wywołania serwisu serializacji mapy: {e}")
    
    def _handle_serialize_response(self, future) -> None:
        """
        Obsługuje odpowiedź od serwisu serializacji mapy SLAM.
        
        Args:
            future: Obiekt future z odpowiedzią serwisu
        """
        try:
            response = future.result()
            if hasattr(response, 'result') and response.result:
                self.get_logger().info("Mapa SLAM została pomyślnie zserializowana")
            else:
                self.get_logger().warning("Serializacja mapy SLAM mogła się nie udać")
        except Exception as e:
            self.get_logger().error(f"Błąd obsługi odpowiedzi serializacji: {e}")
    
    def _handle_save_response(self, future) -> None:
        """
        Obsługuje odpowiedź od serwisu zapisu mapy SLAM.
        
        Args:
            future: Obiekt future z odpowiedzią serwisu
        """
        try:
            response = future.result()
            if hasattr(response, 'result') and response.result:
                self.get_logger().info("Mapa SLAM została pomyślnie zapisana")
            else:
                self.get_logger().warning("Zapis mapy SLAM mogła się nie udać")
        except Exception as e:
            self.get_logger().error(f"Błąd obsługi odpowiedzi zapisu mapy: {e}")
    
    def _save_graph_periodically(self) -> None:
        """
        Okresowo zapisuje wykres postępu eksploracji do pliku.
        Timer callback wykonywany co określony czas dla monitorowania live.
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.MAP_SAVE_DIR, f"live_graph_{timestamp}.png")
            self.graph.save_plot(filename)
        except Exception as e:
            self.get_logger().error(f"Błąd zapisu live wykresu: {e}")
        return None

    def _clear_all_markers(self) -> None:
        """
        Usuwa wszystkie markery wizualizacji z RViz.
        Iteruje przez wszystkich dostępnych publisherów i wysyła polecenie DELETEALL.
        """
        if not self.ENABLE_VISUALIZATION:
            return
            
        publishers = []
        if hasattr(self, 'marker_pub'):
            publishers.append(self.marker_pub)
        if hasattr(self, 'blacklist_pub'):
            publishers.append(self.blacklist_pub)
        if hasattr(self, 'info_pub'):
            publishers.append(self.info_pub)
            
        for pub in publishers:
            marker_array = MarkerArray()
            marker_array.markers = []  # Explicitly initialize as empty list
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            pub.publish(marker_array)
    
    def destroy_node(self):
        """
        Czyszczenie zasobów przed zniszczeniem węzła.
        Zatrzymuje timery, wątki, benchmark i zapisuje finalne wyniki.
        """
        
        # BLOK: Zatrzymywanie wątku wykrywania frontierów
        if self._frontier_thread and self._frontier_thread.is_alive():
            method_name = "FFD" if getattr(self, 'ENABLE_FFD', True) else "WFD"
            self.get_logger().info(f"Zatrzymywanie wątku {method_name}...")
            self._shutdown_event.set()
            self._frontier_thread.join(timeout=2.0)
            
        
        # BLOK: Zatrzymywanie benchmarku i zapisywanie wyników
        if self.ENABLE_BENCHMARK and self.benchmark:
            self.benchmark.stop_benchmark()
            
            # BLOK: Pełna analiza wyników benchmarku
            if self.analyzer:
                results_dir = self.MAP_SAVE_DIR+"/benchmark_results"
                
                # Save results as JSON
                json_path = os.path.join(results_dir, 'benchmark_results.json')
                self.analyzer.save_results_to_file(json_path)
                
                # Create performance plots
                self.analyzer.create_performance_plots(results_dir)
                
                # Create resource usage plots
                self.analyzer.create_resource_plots(results_dir)
                
                # Print summary report
                self.analyzer.print_summary_report()
                
                # Export data as CSV for further analysis
                self.analyzer.export_csv_data(results_dir)
                
                self.get_logger().info(f"Pełna analiza zapisana w: {results_dir}")
        sys.exit(0)
        rclpy.shutdown()
        super().destroy_node()
        

    
def main(args=None):
    """
    Główna funkcja uruchamiająca węzeł eksploracji frontierów.
    Inicjalizuje ROS2, tworzy węzeł i zarządza jego cyklem życia.
    
    Args:
        args: Argumenty linii poleceń ROS2
    """
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Błąd w głównym cyklu eksploracji: {e}")
    finally:
        node.get_logger().info("Zamykanie węzła eksploracji...")
        node._finish_exploration_and_save_map()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

