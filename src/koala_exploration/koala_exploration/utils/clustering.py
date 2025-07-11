from typing import List, Tuple, Set, Optional, Dict
from rclpy.node import Node
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def get_clusters_no(frontiers: List[Tuple[int,int]], 
                                 divisor, min_frontiers, max_frontiers) -> int:
    """
    Oblicza optymalną liczbę klastrów na podstawie liczby frontierów.
    NAPRAWIONA WERSJA: Zapewnia zwracanie typu int z debugging.
    """
    if len(frontiers) == 0:
        return int(min_frontiers)
    
    try:
        # Konwertuj wszystkie parametry na int żeby uniknąć problemów z float
        divisor = int(divisor) if divisor > 0 else 1
        min_frontiers = int(min_frontiers)
        max_frontiers = int(max_frontiers)
        
        # Użyj dzielenia całkowitoliczbowego i jawnej konwersji na int
        calculated_clusters = len(frontiers) // divisor
        result = min(max(min_frontiers, calculated_clusters), max_frontiers)
        
        return int(result)
        
    except Exception as e:
        print(f"ERROR in get_clusters_no: {e}, frontiers={len(frontiers)}, divisor={divisor}, min={min_frontiers}, max={max_frontiers}")
        # Fallback: zwróć bezpieczną wartość
        return max(1, int(min_frontiers) if isinstance(min_frontiers, (int, float)) else 3)
        
        
def _clustering_frontiers_Kmeans(parent: Optional [Node], debug_logging: bool,frontiers: List[Tuple[int,int]], 
                                 divisor, min_frontiers, max_frontiers) -> List[List[Tuple[int, int]]]:
    """
    Grupuje punkty frontierowe w klastry przy użyciu KMeans.
    NAPRAWIONA WERSJA: Lepsze obsługiwanie typów i edge cases.
    
    Args:
        parent: Node dla logowania (opcjonalny)
        debug_logging: Czy włączyć debugging
        frontiers: Lista punktów frontierów jako (row, col)
        divisor: Współczynnik dzielący liczbę punktów
        min_frontiers: Minimalna liczba klastrów
        max_frontiers: Maksymalna liczba klastrów
        
    Returns:
        Lista list punktów reprezentujących klastry
    """
    if not frontiers:
        return []
        
    if len(frontiers) < 2:
        return [frontiers]
    
    # Sprawdź czy parametry są prawidłowe
    if not isinstance(divisor, (int, float)) or divisor <= 0:
        divisor = 100
    if not isinstance(min_frontiers, (int, float)) or min_frontiers < 1:
        min_frontiers = 3
    if not isinstance(max_frontiers, (int, float)) or max_frontiers < min_frontiers:
        max_frontiers = max(15, min_frontiers)
        
    # Konwertuj listę krotek na tablicę NumPy
    points = np.array(frontiers)
    
    # Określ optymalną liczbę klastrów
    num_clusters = get_clusters_no(frontiers, divisor, min_frontiers, max_frontiers)
    
    try:
        # Zapewnij że num_clusters jest typu int z dodatkowym debuggingiem
        if debug_logging and parent:
            parent.get_logger().info(f"KMeans: num_clusters before conversion: {num_clusters}, type: {type(num_clusters)}")
        
        # Zawsze wymuś typ int, nawet jeśli to numpy int lub float
        num_clusters = int(float(num_clusters))  # podwójna konwersja dla pewności
        
        # Sprawdź czy liczba klastrów jest rozsądna
        if num_clusters <= 0:
            num_clusters = 1
        elif num_clusters > len(frontiers):
            num_clusters = len(frontiers)
            
        if debug_logging and parent:
            parent.get_logger().info(f"KMeans: final num_clusters: {num_clusters}, type: {type(num_clusters)}")
            
        # Użyj KMeans do klastryzacji - dodatkowo sprawdź typ tuż przed wywołaniem
        assert isinstance(num_clusters, int), f"num_clusters must be int, got {type(num_clusters)}: {num_clusters}"
        model = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(points)
        labels = model.labels_
        
        # Pogrupuj punkty według etykiet
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(tuple(points[i]))
            
        # Usuń puste klastry
        clusters = [cluster for cluster in clusters if cluster]

        return clusters
        
    except Exception as e:
        error_msg = f"KMeans clustering failed: {e} (num_clusters={num_clusters}, type={type(num_clusters)}, frontiers={len(frontiers)})"
        if parent:
            parent.get_logger().error(f"==================== {error_msg}")
        else:
            print(error_msg)
        # Fallback: jeden klaster zawierający wszystkie punkty
        return [frontiers]

def _clustering_frontiers_DBScan(debug_logging: bool,frontiers: List[Tuple[int,int]], 
                                 divisor, min_frontiers, max_frontiers) -> List[List[Tuple[int, int]]]:
    """
    Grupuje punkty frontierowe w klastry przy użyciu DBSCAN.
    
    Args:
        frontiers: Lista punktów frontierów jako (row, col)
        divisor: Współczynnik dzielący liczbę punktów na klastry
        min_frontiers: Minimalna liczba frontów do utworzenia klastra
        max_frontiers: Maksymalna liczba frontów do utworzenia klastra
        
    Returns:
        Lista list punktów reprezentujących klastry
    """
    if not frontiers:
        return []
        
    if len(frontiers) < 2:
        return [frontiers]
        
    # Konwertuj listę krotek na tablicę NumPy
    points = np.array(frontiers)
    
    # Parametry DBSCAN - dostosuj eps w zależności od skali danych
    eps_values = [5.0]
    min_samples = max(2, min(5, len(frontiers) // 10))
    
    best_clusters = []
    best_score = -1
    
    for eps in eps_values:
        try:
            model = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = model.labels_
            
            # Znajdź unikalne etykiety (pomijając -1 dla szumu)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            if len(unique_labels) == 0:
                continue
                
            # Pogrupuj punkty według etykiet
            clusters = [[] for _ in range(len(unique_labels))]
            label_to_index = {label: i for i, label in enumerate(sorted(unique_labels))}
            
            for i, label in enumerate(labels):
                if label != -1:  # Pomiń punkty oznaczone jako szum
                    clusters[label_to_index[label]].append(tuple(points[i]))
            
            # Usuń puste klastry
            clusters = [cluster for cluster in clusters if cluster]
            
            if debug_logging:
                noise_points = sum(1 for label in labels if label == -1)
                print(f"DBSCAN eps={eps}: {len(clusters)} klastrów, {noise_points} punktów szumu")
            
            # Oceń jakość klastryzacji (liczba klastrów vs punkty szumu)
            score = len(clusters) - (sum(1 for label in labels if label == -1) / len(labels))
            if score > best_score and len(clusters) > 0:
                best_score = score
                best_clusters = clusters
                
        except Exception as e:
            if debug_logging:
                print(f"DBSCAN eps={eps} failed: {e}")
            continue
    
    if not best_clusters:
        # Fallback: jeden klaster zawierający wszystkie punkty
        return [frontiers]
        
    return best_clusters

def _clustering_frontiers_minibatch(debug_logging: bool,frontiers: List[Tuple[int,int]], 
                                 divisor, min_frontiers, max_frontiers) -> List[List[Tuple[int, int]]]:
    """
    Grupuje punkty frontierowe w klastry przy użyciu KMeans.
    
    Args:
        frontiers: Lista punktów frontierów jako (row, col)
        divisor: Współczynnik dzielący liczbę punktów na klastry
        min_frontiers: Minimalna liczba frontów do utworzenia klastra
        max_frontiers: Maksymalna liczba frontów do utworzenia klastra
        
    Returns:
        Lista list punktów reprezentujących klastry
    """
    if not frontiers:
        return []
        
    if len(frontiers) < 2:
        return [frontiers]
        
    # Konwertuj listę krotek na tablicę NumPy
    points = np.array(frontiers)
    
    # Określ optymalną liczbę klastrów (max 10)
    num_clusters = get_clusters_no(frontiers, divisor, min_frontiers, max_frontiers)
    
    try:
        # Zapewnij że num_clusters jest typu int z dodatkowym debuggingiem
        if debug_logging:
            print(f"MiniBatchKMeans: num_clusters before conversion: {num_clusters}, type: {type(num_clusters)}")
        
        # Zawsze wymuś typ int, nawet jeśli to numpy int lub float
        num_clusters = int(float(num_clusters))  # podwójna konwersja dla pewności
        
        # Sprawdź czy liczba klastrów jest rozsądna
        if num_clusters <= 0:
            num_clusters = 1
        elif num_clusters > len(frontiers):
            num_clusters = len(frontiers)
            
        if debug_logging:
            print(f"MiniBatchKMeans: final num_clusters: {num_clusters}, type: {type(num_clusters)}")
            
        # Użyj MiniBatchKMeans do klastryzacji - dodatkowo sprawdź typ tuż przed wywołaniem
        assert isinstance(num_clusters, int), f"num_clusters must be int, got {type(num_clusters)}: {num_clusters}"
        model = MiniBatchKMeans(n_clusters=num_clusters, init="k-means++", random_state=42).fit(points)
        labels = model.labels_
        
        # Pogrupuj punkty według etykiet
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(tuple(points[i]))
            
        # Usuń puste klastry
        clusters = [cluster for cluster in clusters if cluster]
            
        return clusters
        
    except Exception as e:
        if debug_logging:
            print(f"MiniBatchKMeans clustering failed: {e} (num_clusters={num_clusters}, type={type(num_clusters)}, frontiers={len(frontiers)})")
        # Fallback: jeden klaster zawierający wszystkie punkty
        return [frontiers]
    
    
    