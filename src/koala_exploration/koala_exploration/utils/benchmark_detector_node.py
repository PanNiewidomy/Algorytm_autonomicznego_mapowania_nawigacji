#!/usr/bin/env python3
import time
import threading
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import json
import os
from koala_intefaces.msg import Frontiers
import hashlib

@dataclass
class BenchmarkResult:
    method: str
    execution_time: float
    frontier_count: int
    timestamp: float
    frontier_hash: str = ""  # Add a hash field to track unique frontier sets

class ParallelFrontierBenchmark:
    """Równoległy benchmark FFD vs WFD"""
    
    def __init__(self, parent_node, max_results=5000):
        self.parent_node = parent_node
        self.max_results = max_results
        
        # Kolejki wyników
        self.ffd_results = deque(maxlen=max_results)
        self.wfd_results = deque(maxlen=max_results)

        
        # Synchronizacja
        self._map_queue = deque(maxlen=5)
        self._map_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Subskrypcje na wiadomości frontierów
        self.ffd_sub = parent_node.create_subscription(
            Frontiers,
            'FFD/frontiers',
            self._ffd_callback,
            10
        )
        
        self.wfd_sub = parent_node.create_subscription(
            Frontiers,
            'WFD/frontiers',
            self._wfd_callback,
            10
        )
        
        # Statystyki
        self.stats = {
            'ffd_total_time': 0.0,
            'wfd_total_time': 0.0,
            'ffd_count': 0,
            'wfd_count': 0,
            'ffd_avg_frontiers': 0.0,
            'wfd_avg_frontiers': 0.0,
        }
        
        # Data verification
        self._prev_ffd_hash = None
        self._prev_wfd_hash = None
        
        # Resource tracking
        self.ffd_cpu_usage = deque(maxlen=max_results)
        self.wfd_cpu_usage = deque(maxlen=max_results)
        self.ffd_memory_usage = deque(maxlen=max_results)
        self.wfd_memory_usage = deque(maxlen=max_results)
        
    def _compute_frontiers_hash(self, frontiers):
        """Oblicza hash z listy frontierów do sprawdzenia czy dane się zmieniły"""
        frontiers_str = str(sorted([(f.x, f.y) for f in frontiers]))
        return hashlib.md5(frontiers_str.encode()).hexdigest()
    
    def _ffd_callback(self, msg: Frontiers):
        """Callback dla wiadomości z FFD"""
        try:
            # Pomijaj czasy wykonania równe 0
            if msg.cpu_time <= 0.0:
                return
            
            # Oblicz hash frontierów do sprawdzania zmian
            frontiers_hash = self._compute_frontiers_hash(msg.frontiers)
            
            # Sprawdź czy dane się zmieniły
            data_changed = frontiers_hash != self._prev_ffd_hash
            self._prev_ffd_hash = frontiers_hash
            
            # Tylko gdy dane się zmieniły, rejestruj wynik
            if data_changed:
                # Utwórz wynik benchmarku z wiadomości
                result = BenchmarkResult(
                    method="FFD",
                    execution_time=msg.cpu_time,
                    frontier_count=len(msg.frontiers),
                    timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    frontier_hash=frontiers_hash
                )
                
                self.ffd_results.append(result)
                self._update_stats("ffd", result)
                
                # Zapisz dane o zużyciu zasobów
                if hasattr(msg, 'cpu_usage'):
                    self.ffd_cpu_usage.append(msg.cpu_usage)
                if hasattr(msg, 'memory_usage'):
                    self.ffd_memory_usage.append(msg.memory_usage)
            
        except Exception as e:
            self.parent_node.get_logger().error(f"❌ Błąd w FFD benchmark callback: {e}")
    
    def _wfd_callback(self, msg: Frontiers):
        """Callback dla wiadomości z WFD"""
        try:
            # Pomijaj czasy wykonania równe 0
            if msg.cpu_time <= 0.0:
                return
            
            # Oblicz hash frontierów do sprawdzania zmian
            frontiers_hash = self._compute_frontiers_hash(msg.frontiers)
            
            # Sprawdź czy dane się zmieniły
            data_changed = frontiers_hash != self._prev_wfd_hash
            self._prev_wfd_hash = frontiers_hash
            
            # Tylko gdy dane się zmieniły, rejestruj wynik
            if data_changed:
                # Utwórz wynik benchmarku z wiadomości
                result = BenchmarkResult(
                    method="WFD",
                    execution_time=msg.cpu_time,
                    frontier_count=len(msg.frontiers),
                    timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    frontier_hash=frontiers_hash
                )
                
                self.wfd_results.append(result)
                self._update_stats("wfd", result)
                
                # Zapisz dane o zużyciu zasobów
                if hasattr(msg, 'cpu_usage'):
                    self.wfd_cpu_usage.append(msg.cpu_usage)
                if hasattr(msg, 'memory_usage'):
                    self.wfd_memory_usage.append(msg.memory_usage)
            
        except Exception as e:
            self.parent_node.get_logger().error(f"❌ Błąd w WFD benchmark callback: {e}")
    
    def start_benchmark(self):
        self.parent_node.get_logger().info("🏁 Benchmark równoległy FFD vs WFD rozpoczęty")
    
    def add_map_for_benchmark(self, occupancy_grid: OccupancyGrid):
        """Dodaje mapę do benchmarku"""
        with self._map_lock:
            self._map_queue.append(occupancy_grid)
    
    def _update_stats(self, method: str, result: BenchmarkResult):
        """Aktualizuje statystyki"""
        if method == "ffd":
            self.stats['ffd_total_time'] += result.execution_time
            self.stats['ffd_count'] += 1
            self.stats['ffd_avg_frontiers'] = (
                (self.stats['ffd_avg_frontiers'] * (self.stats['ffd_count'] - 1) + result.frontier_count) 
                / self.stats['ffd_count']
            )
        elif method == "wfd":
            self.stats['wfd_total_time'] += result.execution_time
            self.stats['wfd_count'] += 1
            self.stats['wfd_avg_frontiers'] = (
                (self.stats['wfd_avg_frontiers'] * (self.stats['wfd_count'] - 1) + result.frontier_count) 
                / self.stats['wfd_count']
            )
    def get_performance_comparison(self) -> Dict:
        """Zwraca porównanie wydajności"""
        if not self.ffd_results and not self.wfd_results:
            return {}
        
        comparison = {}
        
        # FFD stats
        if self.ffd_results:
            comparison.update({
                'ffd_avg_time': np.mean([r.execution_time for r in self.ffd_results]),
                'ffd_min_time': np.min([r.execution_time for r in self.ffd_results]),
                'ffd_max_time': np.max([r.execution_time for r in self.ffd_results]),
                'ffd_avg_frontiers': np.mean([r.frontier_count for r in self.ffd_results]),
                'ffd_samples': len(self.ffd_results)
            })
        
        # WFD stats
        if self.wfd_results:
            comparison.update({
                'wfd_avg_time': np.mean([r.execution_time for r in self.wfd_results]),
                'wfd_min_time': np.min([r.execution_time for r in self.wfd_results]),
                'wfd_max_time': np.max([r.execution_time for r in self.wfd_results]),
                'wfd_avg_frontiers': np.mean([r.frontier_count for r in self.wfd_results]),
                'wfd_samples': len(self.wfd_results)
            })
        
        # Speedup ratios
        if self.ffd_results and self.wfd_results:
            comparison['wfd_ffd_speedup'] = np.mean([r.execution_time for r in self.wfd_results]) / np.mean([r.execution_time for r in self.ffd_results]) if np.mean([r.execution_time for r in self.ffd_results]) > 0 else 0

        
        return comparison
    
    def stop_benchmark(self):
        """Zatrzymuje benchmark"""
        self.parent_node.get_logger().info("🏁 Benchmark równoległy FFD vs WFD zatrzymany")

class BenchmarkAnalyzer:
    """Analiza i wizualizacja wyników benchmarku"""
    
    def __init__(self, benchmark_instance):
        self.benchmark = benchmark_instance
    
    def save_results_to_file(self, filename: str):
        """Zapisuje wyniki do pliku JSON"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = {
            'ffd_results': [
                {
                    'execution_time': r.execution_time,
                    'frontier_count': r.frontier_count,
                    'timestamp': r.timestamp
                }
                for r in self.benchmark.ffd_results
            ],
            'wfd_results': [
                {
                    'execution_time': r.execution_time,
                    'frontier_count': r.frontier_count,
                    'timestamp': r.timestamp
                }
                for r in self.benchmark.wfd_results
            ],
            'stats': self.benchmark.stats,
            'comparison': self.benchmark.get_performance_comparison()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Wyniki benchmarku zapisane do: {filename}")
    
    def create_performance_plots(self, save_dir: str):
        """Tworzy wykresy wydajności"""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.benchmark.ffd_results and not self.benchmark.wfd_results:
            print("❌ Brak wyników do analizy")
            return
        
        # Przygotuj dane
        ffd_times = [r.execution_time * 1000 for r in self.benchmark.ffd_results] if self.benchmark.ffd_results else []
        wfd_times = [r.execution_time * 1000 for r in self.benchmark.wfd_results] if self.benchmark.wfd_results else []
        ffd_frontiers = [r.frontier_count for r in self.benchmark.ffd_results] if self.benchmark.ffd_results else []
        wfd_frontiers = [r.frontier_count for r in self.benchmark.wfd_results] if self.benchmark.wfd_results else []
        
        # Twórz wykresy
        plt.figure(figsize=(18, 12))
        
        # 1. Porównanie czasów wykonania (Box Plot)
        plt.subplot(3, 3, 1)
        data_to_plot = []
        labels = []
        
        if ffd_times:
            data_to_plot.append(ffd_times)
            labels.append('FFD')
        if wfd_times:
            data_to_plot.append(wfd_times)
            labels.append('WFD')
            
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Czas wykonania (ms)')
            plt.title('Porównanie czasów wykonania')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        # 2. Histogram czasów wykonania
        plt.subplot(3, 3, 2)
        if any([ffd_times, wfd_times,]):
            all_times = ffd_times + wfd_times
            bins = min(20, max(5, len(all_times) // 5)) if all_times else 10
            
            if ffd_times:
                plt.hist(ffd_times, bins=bins, alpha=0.7, label=f'FFD (n={len(ffd_times)})', 
                        color='blue', density=True)
            if wfd_times:
                plt.hist(wfd_times, bins=bins, alpha=0.7, label=f'WFD (n={len(wfd_times)})', 
                        color='orange', density=True)
            
            plt.xlabel('Czas wykonania (ms)')
            plt.ylabel('Gęstość')
            plt.title('Rozkład czasów wykonania')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Liczba frontierów w czasie
        plt.subplot(3, 3, 3)
        if ffd_frontiers:
            iterations_ffd = range(len(ffd_frontiers))
            plt.plot(iterations_ffd, ffd_frontiers, label=f'FFD (avg: {np.mean(ffd_frontiers):.1f})', 
                    alpha=0.8, linewidth=1, color='blue')
        if wfd_frontiers:
            iterations_wfd = range(len(wfd_frontiers))
            plt.plot(iterations_wfd, wfd_frontiers, label=f'WFD (avg: {np.mean(wfd_frontiers):.1f})', 
                    alpha=0.8, linewidth=1, color='orange')
        
        plt.ylabel('Liczba frontierów')
        plt.xlabel('Iteracja')
        plt.title('Liczba frontierów w czasie')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Korelacja frontierów vs czas
        plt.subplot(3, 3, 4)
        if ffd_frontiers and ffd_times:
            plt.scatter(ffd_frontiers, ffd_times, label=f'FFD (r={np.corrcoef(ffd_frontiers, ffd_times)[0,1]:.3f})', 
                       alpha=0.6, s=30, color='blue')
        if wfd_frontiers and wfd_times:
            plt.scatter(wfd_frontiers, wfd_times, label=f'WFD (r={np.corrcoef(wfd_frontiers, wfd_times)[0,1]:.3f})', 
                       alpha=0.6, s=30, color='orange')
        
        plt.xlabel('Liczba frontierów')
        plt.ylabel('Czas wykonania (ms)')
        plt.title('Korelacja: frontiers vs czas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Wydajność względna (speedup)
        plt.subplot(3, 3, 5)
        comparison = self.benchmark.get_performance_comparison()
        if comparison:
            speedup_data = []
            speedup_labels = []
            
            if 'wfd_ffd_speedup' in comparison and comparison['wfd_ffd_speedup'] > 0:
                speedup_data.append(comparison['wfd_ffd_speedup'])
                speedup_labels.append('WFD/FFD')
           
            if speedup_data:
                colors = ['orange', 'green', 'red'][:len(speedup_data)]
                bars = plt.bar(speedup_labels, speedup_data, color=colors, alpha=0.7)
                plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Equal Performance')
                plt.ylabel('Speedup Ratio')
                plt.title('Względna wydajność (>1 = licznik szybszy)')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # 6. Statystyki porównawcze
        plt.subplot(3, 3, 6)
        comparison = self.benchmark.get_performance_comparison()
        if comparison and len(comparison) > 0:
            metrics = ['Avg Time', 'Min Time', 'Max Time']
            ffd_values = [comparison.get('ffd_avg_time', 0)*1000, 
                         comparison.get('ffd_min_time', 0)*1000, 
                         comparison.get('ffd_max_time', 0)*1000]
            wfd_values = [comparison.get('wfd_avg_time', 0)*1000, 
                         comparison.get('wfd_min_time', 0)*1000, 
                         comparison.get('wfd_max_time', 0)*1000]
           
            x = np.arange(len(metrics))
            width = 0.25
            
            if any(v > 0 for v in ffd_values):
                plt.bar(x - width, ffd_values, width, label='FFD', alpha=0.8, color='blue')
            if any(v > 0 for v in wfd_values):
                plt.bar(x, wfd_values, width, label='WFD', alpha=0.8, color='orange')
            
            plt.ylabel('Czas (ms)')
            plt.title('Porównanie statystyk')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3)
            if max(max(ffd_values), max(wfd_values)) > 0:
                plt.yscale('log')
        
        # 7. CPU Usage Comparison
        plt.subplot(3, 3, 7)
        cpu_data = []
        cpu_labels = []
        
        if self.benchmark.ffd_cpu_usage:
            cpu_data.append(list(self.benchmark.ffd_cpu_usage))
            cpu_labels.append('FFD')
        if self.benchmark.wfd_cpu_usage:
            cpu_data.append(list(self.benchmark.wfd_cpu_usage))
            cpu_labels.append('WFD')
            
        if cpu_data:
            plt.boxplot(cpu_data, labels=cpu_labels)
            plt.ylabel('Zużycie CPU (%)')
            plt.title('Porównanie zużycia CPU')
            plt.grid(True, alpha=0.3)
        
        # 8. Memory Usage Comparison
        plt.subplot(3, 3, 8)
        memory_data = []
        memory_labels = []
        
        if self.benchmark.ffd_memory_usage:
            memory_data.append(list(self.benchmark.ffd_memory_usage))
            memory_labels.append('FFD')
        if self.benchmark.wfd_memory_usage:
            memory_data.append(list(self.benchmark.wfd_memory_usage))
            memory_labels.append('WFD')

        if memory_data:
            plt.boxplot(memory_data, labels=memory_labels)
            plt.ylabel('Zużycie pamięci (MB)')
            plt.title('Porównanie zużycia pamięci')
            plt.grid(True, alpha=0.3)
        
        # 9. Summary Performance Bar Chart
        plt.subplot(3, 3, 9)
        if comparison:
            methods = []
            avg_times = []
            colors = []
            
            if 'ffd_avg_time' in comparison:
                methods.append('FFD')
                avg_times.append(comparison['ffd_avg_time'] * 1000)
                colors.append('blue')
            if 'wfd_avg_time' in comparison:
                methods.append('WFD')
                avg_times.append(comparison['wfd_avg_time'] * 1000)
                colors.append('orange')
            
            if methods and avg_times:
                bars = plt.bar(methods, avg_times, color=colors, alpha=0.8)
                plt.ylabel('Średni czas (ms)')
                plt.title('Podsumowanie wydajności')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, avg_times):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}ms',
                            ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Zapisz wykresy
        plot_path = os.path.join(save_dir, 'benchmark_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Wykresy analizy zapisane do: {plot_path}")
        
        # Dodatkowo: wykres czasowy
        self._create_timeline_plot(save_dir)
    
    def _create_timeline_plot(self, save_dir: str):
        """Tworzy wykres czasowy wydajności"""
        plt.figure(figsize=(12, 8))
        
        if not self.benchmark.ffd_results and not self.benchmark.wfd_results:
            print("❌ Brak wyników do wykresu czasowego")
            return
        
        # Inicjalizuj referencyjny timestamp
        ref_timestamp = None
        if self.benchmark.ffd_results:
            ref_timestamp = self.benchmark.ffd_results[0].timestamp
        elif self.benchmark.wfd_results:
            ref_timestamp = self.benchmark.wfd_results[0].timestamp
        
        if ref_timestamp is None:
            print("❌ Brak timestampów do wykresu czasowego")
            return
        
        # Przygotuj dane
        ffd_data = []
        wfd_data = []

        if self.benchmark.ffd_results:
            ffd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.ffd_results]
            ffd_times = [r.execution_time * 1000 for r in self.benchmark.ffd_results]
            ffd_data = (ffd_timestamps, ffd_times)
        
        if self.benchmark.wfd_results:
            wfd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.wfd_results]
            wfd_times = [r.execution_time * 1000 for r in self.benchmark.wfd_results]
            wfd_data = (wfd_timestamps, wfd_times)
        
        plt.subplot(2, 1, 1)
        if ffd_data:
            plt.plot(ffd_data[0], ffd_data[1], 'o-', label=f'FFD (n={len(ffd_data[1])})', 
                    alpha=0.7, markersize=3, color='blue')
        if wfd_data:
            plt.plot(wfd_data[0], wfd_data[1], 's-', label=f'WFD (n={len(wfd_data[1])})', 
                    alpha=0.7, markersize=3, color='orange')
            
        plt.ylabel('Czas wykonania (ms)')
        plt.title('Wydajność w czasie')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(2, 1, 2)
        # Moving average (okno 10 próbek)
        if ffd_data and len(ffd_data[1]) >= 10:
            ffd_ma = np.convolve(ffd_data[1], np.ones(10)/10, mode='valid')
            ffd_ma_timestamps = ffd_data[0][9:]  # Dopasuj długość
            plt.plot(ffd_ma_timestamps, ffd_ma, '-', label='FFD (średnia ruchoma)', 
                    linewidth=2, color='blue')
        
        if wfd_data and len(wfd_data[1]) >= 10:
            wfd_ma = np.convolve(wfd_data[1], np.ones(10)/10, mode='valid')
            wfd_ma_timestamps = wfd_data[0][9:]  # Dopasuj długość
            plt.plot(wfd_ma_timestamps, wfd_ma, '-', label='WFD (średnia ruchoma)', 
                    linewidth=2, color='orange')
        
        
        plt.xlabel('Czas (minuty)')
        plt.ylabel('Czas wykonania (ms)')
        plt.title('Trend wydajności (średnia ruchoma)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        timeline_path = os.path.join(save_dir, 'benchmark_timeline.png')
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⏱️ Wykres czasowy zapisany do: {timeline_path}")
    
    def print_summary_report(self):
        """Drukuje podsumowanie benchmarku"""
        comparison = self.benchmark.get_performance_comparison()
        if not comparison:
            print("❌ Brak danych do raportu")
            return
        
        print("\n" + "="*70)
        print("📊 PODSUMOWANIE BENCHMARKU FFD vs WFD")
        print("="*70)
        
        print(f"📈 Liczba próbek:")
        if 'ffd_samples' in comparison:
            print(f"   FFD: {comparison['ffd_samples']}")
        if 'wfd_samples' in comparison:
            print(f"   WFD: {comparison['wfd_samples']}")

        print(f"\n⏱️ Średni czas wykonania:")
        if 'ffd_avg_time' in comparison:
            print(f"   FFD: {comparison['ffd_avg_time']*1000:.2f} ms")
        if 'wfd_avg_time' in comparison:
            print(f"   WFD: {comparison['wfd_avg_time']*1000:.2f} ms")

        print(f"\n⚡ Zakres czasów:")
        if 'ffd_min_time' in comparison and 'ffd_max_time' in comparison:
            print(f"   FFD: {comparison['ffd_min_time']*1000:.2f} - {comparison['ffd_max_time']*1000:.2f} ms")
        if 'wfd_min_time' in comparison and 'wfd_max_time' in comparison:
            print(f"   WFD: {comparison['wfd_min_time']*1000:.2f} - {comparison['wfd_max_time']*1000:.2f} ms")

        print(f"\n🎯 Średnia liczba frontierów:")
        if 'ffd_avg_frontiers' in comparison:
            print(f"   FFD: {comparison['ffd_avg_frontiers']:.1f}")
        if 'wfd_avg_frontiers' in comparison:
            print(f"   WFD: {comparison['wfd_avg_frontiers']:.1f}")

        
        # Add resource usage reporting
        cpu_metrics = ['ffd_avg_cpu', 'wfd_avg_cpu']
        if any(metric in comparison for metric in cpu_metrics):
            print(f"\n💻 Średnie zużycie CPU:")
            if 'ffd_avg_cpu' in comparison:
                print(f"   FFD: {comparison['ffd_avg_cpu']:.2f}%")
            if 'wfd_avg_cpu' in comparison:
                print(f"   WFD: {comparison['wfd_avg_cpu']:.2f}%")

    
        memory_metrics = ['ffd_avg_memory', 'wfd_avg_memory']
        if any(metric in comparison for metric in memory_metrics):
            print(f"\n🧠 Średnie zużycie pamięci:")
            if 'ffd_avg_memory' in comparison:
                print(f"   FFD: {comparison['ffd_avg_memory']:.2f} MB")
            if 'wfd_avg_memory' in comparison:
                print(f"   WFD: {comparison['wfd_avg_memory']:.2f} MB")

        print(f"\n🏆 Porównania wydajności:")
        if 'wfd_ffd_speedup' in comparison:
            speedup = comparison['wfd_ffd_speedup']
            if speedup > 1.0:
                print(f"   FFD jest {speedup:.2f}x SZYBSZY niż WFD")
            elif speedup < 1.0:
                print(f"   WFD jest {1/speedup:.2f}x SZYBSZY niż FFD")
            else:
                print(f"   FFD i WFD mają porównywalną wydajność")
        
        
        # Determine overall winner
        avg_times = {}
        if 'ffd_avg_time' in comparison:
            avg_times['FFD'] = comparison['ffd_avg_time']
        if 'wfd_avg_time' in comparison:
            avg_times['WFD'] = comparison['wfd_avg_time']
        if avg_times:
            
            winner = min(avg_times.keys(), key=lambda k: avg_times[k])
            print(f"\n🥇 ZWYCIĘZCA OGÓLNY: {winner}")
        
        print("="*70)
    
    def export_csv_data(self, save_dir: str):
        """Eksportuje dane do plików CSV"""
        import csv
        
        os.makedirs(save_dir, exist_ok=True)
        
        # FFD data
        if self.benchmark.ffd_results:
            ffd_path = os.path.join(save_dir, 'ffd_results.csv')
            with open(ffd_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'execution_time_ms', 'frontier_count'])
                for r in self.benchmark.ffd_results:
                    writer.writerow([r.timestamp, r.execution_time*1000, r.frontier_count])
        
        # WFD data
        if self.benchmark.wfd_results:
            wfd_path = os.path.join(save_dir, 'wfd_results.csv')
            with open(wfd_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'execution_time_ms', 'frontier_count'])
                for r in self.benchmark.wfd_results:
                    writer.writerow([r.timestamp, r.execution_time*1000, r.frontier_count])
        
        print(f"📄 Dane CSV zapisane do: {save_dir}")
    
    def create_resource_plots(self, save_dir: str):
        """Tworzy wykresy zużycia zasobów"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have resource data
        has_cpu_data = bool(self.benchmark.ffd_cpu_usage or self.benchmark.wfd_cpu_usage )
        has_memory_data = bool(self.benchmark.ffd_memory_usage or self.benchmark.wfd_memory_usage)
        
        if not has_cpu_data and not has_memory_data:
            print("❌ Brak danych o zasobach do analizy")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. CPU Usage Box Plot
        plt.subplot(2, 3, 1)
        data_to_plot = []
        labels = []
        
        if self.benchmark.ffd_cpu_usage:
            data_to_plot.append(list(self.benchmark.ffd_cpu_usage))
            labels.append('FFD')
        if self.benchmark.wfd_cpu_usage:
            data_to_plot.append(list(self.benchmark.wfd_cpu_usage))
            labels.append('WFD')
            
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Zużycie CPU (%)')
            plt.title('Porównanie zużycia CPU')
            plt.grid(True, alpha=0.3)
        
        # 2. Memory Usage Box Plot
        plt.subplot(2, 3, 2)
        data_to_plot = []
        labels = []
        
        if self.benchmark.ffd_memory_usage:
            data_to_plot.append(list(self.benchmark.ffd_memory_usage))
            labels.append('FFD')
        if self.benchmark.wfd_memory_usage:
            data_to_plot.append(list(self.benchmark.wfd_memory_usage))
            labels.append('WFD')

        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Zużycie pamięci (MB)')
            plt.title('Porównanie zużycia pamięci')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Zapisz wykresy
        plot_path = os.path.join(save_dir, 'resource_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Wykresy zużycia zasobów zapisane do: {plot_path}")
        
        # Dodatkowo: wykres czasowy zasobów
        self._create_resource_timeline_plot(save_dir)

    def _create_resource_timeline_plot(self, save_dir: str):
        """Tworzy wykres czasowy zużycia zasobów"""
        plt.figure(figsize=(12, 10))
        
        has_cpu_data = bool(self.benchmark.ffd_cpu_usage or self.benchmark.wfd_cpu_usage)
        has_memory_data = bool(self.benchmark.ffd_memory_usage or self.benchmark.wfd_memory_usage)
        
        if not has_cpu_data and not has_memory_data:
            return
        
        # Inicjalizuj referencyjny timestamp
        ref_timestamp = None
        if self.benchmark.ffd_results:
            ref_timestamp = self.benchmark.ffd_results[0].timestamp
        elif self.benchmark.wfd_results:
            ref_timestamp = self.benchmark.wfd_results[0].timestamp

        if ref_timestamp is None:
            return
        
        # CPU Usage Timeline
        if has_cpu_data:
            plt.subplot(2, 1, 1)
            
            if self.benchmark.ffd_cpu_usage and self.benchmark.ffd_results:
                ffd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.ffd_results]
                ffd_cpu = list(self.benchmark.ffd_cpu_usage)
                min_len = min(len(ffd_timestamps), len(ffd_cpu))
                if min_len > 0:
                    plt.plot(ffd_timestamps[:min_len], ffd_cpu[:min_len], 'o-', 
                            label=f'FFD CPU (n={min_len})', alpha=0.7, markersize=3, color='blue')
                    
                    # Moving average for CPU usage
                    if min_len >= 10:
                        window_size = min(10, min_len // 2)
                        ffd_cpu_ma = np.convolve(ffd_cpu[:min_len], np.ones(window_size)/window_size, mode='valid')
                        ffd_cpu_ma_timestamps = ffd_timestamps[:min_len][window_size-1:]
                        plt.plot(ffd_cpu_ma_timestamps, ffd_cpu_ma, '-', 
                                label='FFD CPU (średnia ruchoma)', linewidth=2, color='darkblue')
            
            if self.benchmark.wfd_cpu_usage and self.benchmark.wfd_results:
                wfd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.wfd_results]
                wfd_cpu = list(self.benchmark.wfd_cpu_usage)
                min_len = min(len(wfd_timestamps), len(wfd_cpu))
                if min_len > 0:
                    plt.plot(wfd_timestamps[:min_len], wfd_cpu[:min_len], 's-', 
                            label=f'WFD CPU (n={min_len})', alpha=0.7, markersize=3, color='orange')
                    
                    # Moving average for CPU usage
                    if min_len >= 10:
                        window_size = min(10, min_len // 2)
                        wfd_cpu_ma = np.convolve(wfd_cpu[:min_len], np.ones(window_size)/window_size, mode='valid')
                        wfd_cpu_ma_timestamps = wfd_timestamps[:min_len][window_size-1:]
                        plt.plot(wfd_cpu_ma_timestamps, wfd_cpu_ma, '-', 
                                label='WFD CPU (średnia ruchoma)', linewidth=2, color='darkorange')
            
            
            plt.ylabel('Zużycie CPU (%)')
            plt.title('Zużycie CPU w czasie')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Memory Usage Timeline
        if has_memory_data:
            plt.subplot(2, 1, 2)
            
            if self.benchmark.ffd_memory_usage and self.benchmark.ffd_results:
                ffd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.ffd_results]
                ffd_memory = list(self.benchmark.ffd_memory_usage)
                min_len = min(len(ffd_timestamps), len(ffd_memory))
                if min_len > 0:
                    plt.plot(ffd_timestamps[:min_len], ffd_memory[:min_len], 'o-', 
                            label=f'FFD Pamięć (n={min_len})', alpha=0.7, markersize=3, color='blue')
                    
                    # Moving average for memory usage
                    if min_len >= 10:
                        window_size = min(10, min_len // 2)
                        ffd_memory_ma = np.convolve(ffd_memory[:min_len], np.ones(window_size)/window_size, mode='valid')
                        ffd_memory_ma_timestamps = ffd_timestamps[:min_len][window_size-1:]
                        plt.plot(ffd_memory_ma_timestamps, ffd_memory_ma, '-', 
                                label='FFD Pamięć (średnia ruchoma)', linewidth=2, color='darkblue')
            
            if self.benchmark.wfd_memory_usage and self.benchmark.wfd_results:
                wfd_timestamps = [(r.timestamp - ref_timestamp) / 60 for r in self.benchmark.wfd_results]
                wfd_memory = list(self.benchmark.wfd_memory_usage)
                min_len = min(len(wfd_timestamps), len(wfd_memory))
                if min_len > 0:
                    plt.plot(wfd_timestamps[:min_len], wfd_memory[:min_len], 's-', 
                            label=f'WFD Pamięć (n={min_len})', alpha=0.7, markersize=3, color='orange')
                    
                    # Moving average for memory usage
                    if min_len >= 10:
                        window_size = min(10, min_len // 2)
                        wfd_memory_ma = np.convolve(wfd_memory[:min_len], np.ones(window_size)/window_size, mode='valid')
                        wfd_memory_ma_timestamps = wfd_timestamps[:min_len][window_size-1:]
                        plt.plot(wfd_memory_ma_timestamps, wfd_memory_ma, '-', 
                                label='WFD Pamięć (średnia ruchoma)', linewidth=2, color='darkorange')
                        
            
            plt.xlabel('Czas (minuty)')
            plt.ylabel('Zużycie pamięci (MB)')
            plt.title('Zużycie pamięci w czasie')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        
        resource_timeline_path = os.path.join(save_dir, 'resource_timeline.png')
        plt.savefig(resource_timeline_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⏱️ Wykres czasowy zasobów zapisany do: {resource_timeline_path}")