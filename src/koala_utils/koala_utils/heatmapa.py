import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import os
import argparse

class PGMHeatmapGenerator:
    def __init__(self):
        self.supported_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 
            'spring', 'summer', 'autumn', 'winter', 'jet', 'rainbow'
        ]
    
    def read_pgm(self, filepath):
        """
        Wczytuje plik PGM i zwraca jako numpy array
        """
        try:
            # Próba wczytania za pomocą PIL
            with Image.open(filepath) as img:
                # Konwersja do trybu grayscale jeśli potrzeba
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        except Exception as e:
            print(f"Błąd przy wczytywaniu {filepath} przez PIL: {e}")
            
            # Alternatywna metoda - ręczne parsowanie PGM
            try:
                return self._read_pgm_manual(filepath)
            except Exception as e2:
                print(f"Błąd przy ręcznym parsowaniu {filepath}: {e2}")
                return None
    
    def _read_pgm_manual(self, filepath):
        """
        Ręczne parsowanie pliku PGM
        """
        with open(filepath, 'rb') as f:
            # Czytaj nagłówek
            header = f.readline().decode('ascii').strip()
            
            if header not in ['P2', 'P5']:
                raise ValueError(f"Nieobsługiwany format PGM: {header}")
            
            # Pomiń komentarze
            line = f.readline().decode('ascii').strip()
            while line.startswith('#'):
                line = f.readline().decode('ascii').strip()
            
            # Wymiary
            width, height = map(int, line.split())
            
            # Maksymalna wartość
            max_val = int(f.readline().decode('ascii').strip())
            
            if header == 'P2':  # ASCII format
                data = []
                for line in f:
                    data.extend(map(int, line.decode('ascii').split()))
                data = np.array(data).reshape((height, width))
            else:  # P5 - binary format
                data = np.frombuffer(f.read(), dtype=np.uint8)
                data = data.reshape((height, width))
            
            return data.astype(np.float32)
    
    def create_heatmap(self, data, title="Heatmapa PGM", colormap='viridis', 
                      save_path=None, show_colorbar=True, figsize=(10, 8)):
        """
        Tworzy heatmapę z danych
        """
        plt.figure(figsize=figsize)
        
        # Tworzenie heatmapy
        im = plt.imshow(data, cmap=colormap, aspect='auto')
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        if show_colorbar:
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Intensywność', rotation=270, labelpad=20)
        
        # Usunięcie osi dla czystszego wyglądu
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmapa zapisana jako: {save_path}")
        
        plt.show()
    
    def process_single_file(self, filepath, colormap='viridis', output_dir=None):
        """
        Przetwarza pojedynczy plik PGM
        """
        print(f"Przetwarzanie: {filepath}")
        
        data = self.read_pgm(filepath)
        if data is None:
            print(f"Nie udało się wczytać {filepath}")
            return
        
        filename = os.path.splitext(os.path.basename(filepath))[0]
        title = f"Heatmapa: {filename}"
        
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{filename}_heatmap.png")
        
        self.create_heatmap(data, title=title, colormap=colormap, save_path=save_path)
        
        # Wyświetl statystyki
        print(f"Wymiary: {data.shape}")
        print(f"Min: {data.min():.2f}, Max: {data.max():.2f}, Średnia: {data.mean():.2f}")
        print("-" * 50)
    
    def process_directory(self, directory, colormap='viridis', output_dir=None, recursive=True):
        """
        Przetwarza wszystkie pliki PGM w katalogu (opcjonalnie rekurencyjnie)
        """
        pgm_files = []
        
        if recursive:
            # Przeszukuj rekurencyjnie wszystkie podkatalogi
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.pgm', '.pnm')):
                        pgm_files.append(os.path.join(root, file))
        else:
            # Przeszukuj tylko główny katalog
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if os.path.isfile(filepath) and file.lower().endswith(('.pgm', '.pnm')):
                    pgm_files.append(filepath)
        
        if not pgm_files:
            search_type = "rekurencyjnie" if recursive else ""
            print(f"Nie znaleziono plików PGM w katalogu {search_type}: {directory}")
            return
        
        # Sortuj pliki alfabetycznie dla lepszej organizacji
        pgm_files.sort()
        
        search_info = "rekurencyjnie " if recursive else ""
        print(f"Znaleziono {search_info}{len(pgm_files)} plików PGM:")
        
        # Wyświetl znalezione pliki z ich względnymi ścieżkami
        for i, filepath in enumerate(pgm_files, 1):
            rel_path = os.path.relpath(filepath, directory)
            print(f"  {i}. {rel_path}")
        
        print("-" * 50)
        
        for i, filepath in enumerate(pgm_files, 1):
            print(f"[{i}/{len(pgm_files)}] Przetwarzanie: {os.path.relpath(filepath, directory)}")
            
            # Zachowaj strukturę katalogów w output_dir
            if output_dir:
                rel_path = os.path.relpath(filepath, directory)
                rel_dir = os.path.dirname(rel_path)
                filename = os.path.splitext(os.path.basename(filepath))[0]
                
                final_output_dir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
                os.makedirs(final_output_dir, exist_ok=True)
                save_path = os.path.join(final_output_dir, f"{filename}_heatmap.png")
            else:
                save_path = None
            
            # Przetworz plik
            data = self.read_pgm(filepath)
            if data is None:
                print(f"  ❌ Nie udało się wczytać pliku")
                continue
            
            rel_path = os.path.relpath(filepath, directory)
            filename = os.path.splitext(os.path.basename(filepath))[0]
            title = f"Heatmapa: {rel_path}"
            
            try:
                self.create_heatmap(data, title=title, colormap=colormap, save_path=save_path)
                print(f"  ✅ Wymiary: {data.shape}, Min: {data.min():.2f}, Max: {data.max():.2f}, Średnia: {data.mean():.2f}")
            except Exception as e:
                print(f"  ❌ Błąd podczas tworzenia heatmapy: {e}")
            
            print("-" * 30)
    
    def compare_colormaps(self, filepath, colormaps=None):
        """
        Porównuje różne mapy kolorów dla tego samego pliku
        """
        if colormaps is None:
            colormaps = ['viridis', 'plasma', 'hot', 'cool']
        
        data = self.read_pgm(filepath)
        if data is None:
            print(f"Nie udało się wczytać {filepath}")
            return
        
        filename = os.path.splitext(os.path.basename(filepath))[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, cmap in enumerate(colormaps[:4]):
            im = axes[i].imshow(data, cmap=cmap, aspect='auto')
            axes[i].set_title(f'{filename} - {cmap}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        plt.show()
    
    def overlay_files(self, filepaths, method='mean', normalize=True):
        """
        Nakłada wszystkie pliki PGM na siebie
        
        Args:
            filepaths: Lista ścieżek do plików PGM
            method: 'mean' (średnia), 'sum' (suma), 'max' (maksimum), 'min' (minimum)
            normalize: Czy normalizować wynik do 0-255
        
        Returns:
            numpy array z nałożonymi obrazami lub None w przypadku błędu
        """
        if not filepaths:
            print("Brak plików do nałożenia")
            return None
        
        valid_data = []
        dimensions = {}
        
        print(f"Wczytywanie {len(filepaths)} plików...")
        
        # Wczytaj wszystkie pliki i sprawdź wymiary
        for i, filepath in enumerate(filepaths):
            data = self.read_pgm(filepath)
            if data is not None:
                shape_key = f"{data.shape[0]}x{data.shape[1]}"
                if shape_key not in dimensions:
                    dimensions[shape_key] = []
                dimensions[shape_key].append((filepath, data))
                valid_data.append((filepath, data))
                print(f"  ✅ {os.path.basename(filepath)}: {data.shape}")
            else:
                print(f"  ❌ Nie udało się wczytać: {os.path.basename(filepath)}")
        
        if not valid_data:
            print("Nie udało się wczytać żadnego pliku")
            return None
        
        # Sprawdź czy wszystkie obrazy mają te same wymiary
        if len(dimensions) > 1:
            print(f"\n⚠️  UWAGA: Znaleziono obrazy o różnych wymiarach:")
            for dim, files in dimensions.items():
                print(f"  {dim}: {len(files)} plików")
        
        # Użyj największej grupy obrazów o takich samych wymiarach
        largest_group = max(dimensions.values(), key=len)
        if len(largest_group) < len(valid_data):
            print(f"\n📏 Używam {len(largest_group)} obrazów o wymiarach {largest_group[0][1].shape}")
            print("   Obrazy o innych wymiarach zostały pominięte")
        
        # Przygotuj dane do nałożenia
        overlay_data = [data for _, data in largest_group]
        overlay_files = [filepath for filepath, _ in largest_group]
        
        print(f"\n🔄 Nakładanie {len(overlay_data)} obrazów metodą '{method}'...")
        
        # Stwórz stos obrazów
        image_stack = np.stack(overlay_data, axis=0)
        
        # Zastosuj wybraną metodę nałożenia
        if method == 'mean':
            result = np.mean(image_stack, axis=0)
        elif method == 'sum':
            result = np.sum(image_stack, axis=0)
        elif method == 'max':
            result = np.max(image_stack, axis=0)
        elif method == 'min':
            result = np.min(image_stack, axis=0)
        elif method == 'median':
            result = np.median(image_stack, axis=0)
        elif method == 'std':
            result = np.std(image_stack, axis=0)
        else:
            print(f"Nieznana metoda: {method}. Używam 'mean'")
            result = np.mean(image_stack, axis=0)
        
        # Normalizacja
        if normalize and method != 'std':
            if result.max() > result.min():
                result = ((result - result.min()) / (result.max() - result.min())) * 255
            else:
                result = np.zeros_like(result)
        
        print(f"✅ Nałożono {len(overlay_data)} obrazów")
        print(f"   Wymiary wyniku: {result.shape}")
        print(f"   Zakres wartości: {result.min():.2f} - {result.max():.2f}")
        
        return result, overlay_files
    
    def create_overlay_heatmap(self, directory, method='mean', colormap='viridis', 
                              recursive=True, output_dir=None, show_individual=False):
        """
        Tworzy heatmapę z nałożonych plików PGM z katalogu
        """
        # Znajdź wszystkie pliki PGM
        pgm_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.pgm', '.pnm')):
                        pgm_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if os.path.isfile(filepath) and file.lower().endswith(('.pgm', '.pnm')):
                    pgm_files.append(filepath)
        
        if not pgm_files:
            search_type = "rekurencyjnie" if recursive else ""
            print(f"Nie znaleziono plików PGM w katalogu {search_type}: {directory}")
            return
        
        pgm_files.sort()
        print(f"Znaleziono {len(pgm_files)} plików PGM do nałożenia")
        
        # Nałóż pliki na siebie
        result = self.overlay_files(pgm_files, method=method)
        if result is None:
            return
        
        overlay_data, used_files = result
        
        # Stwórz tytuł
        method_names = {
            'mean': 'Średnia',
            'sum': 'Suma', 
            'max': 'Maksimum',
            'min': 'Minimum',
            'median': 'Mediana',
            'std': 'Odchylenie standardowe'
        }
        method_name = method_names.get(method, method)
        title = f"Heatmapa nałożona ({method_name}) - {len(used_files)} plików"
        
        # Przygotuj ścieżkę zapisu
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dir_name = os.path.basename(directory.rstrip('/'))
            save_path = os.path.join(output_dir, f"nalozenie_{method}_{dir_name}.png")
        
        # Stwórz główną heatmapę
        self.create_heatmap(overlay_data, title=title, colormap=colormap, save_path=save_path)
        
        # Opcjonalnie pokaż pojedyncze pliki dla porównania
        if show_individual:
            self._show_overlay_comparison(used_files, overlay_data, method, colormap)
    
    def _show_overlay_comparison(self, filepaths, overlay_result, method, colormap):
        """
        Pokazuje porównanie pierwszych kilku plików z wynikiem nałożenia
        """
        max_individual = 6  # Maksymalnie 6 pojedynczych obrazów do pokazania
        
        # Wczytaj kilka pierwszych plików
        individual_data = []
        individual_titles = []
        
        for filepath in filepaths[:max_individual]:
            data = self.read_pgm(filepath)
            if data is not None:
                individual_data.append(data)
                individual_titles.append(os.path.basename(filepath))
        
        # Dodaj wynik nałożenia
        individual_data.append(overlay_result)
        individual_titles.append(f'WYNIK ({method})')
        
        # Stwórz porównanie
        n_total = len(individual_data)
        if n_total <= 4:
            rows, cols = 2, 2
        elif n_total <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Znajdź globalną skalę
        all_data = individual_data[:-1]  # Bez wyniku nałożenia
        if all_data:
            global_min = min(data.min() for data in all_data)
            global_max = max(data.max() for data in all_data)
        else:
            global_min, global_max = overlay_result.min(), overlay_result.max()
        
        for i, (data, title) in enumerate(zip(individual_data, individual_titles)):
            ax = axes[i]
            
            # Dla wyniku nałożenia użyj jego własnej skali
            if i == len(individual_data) - 1:  # Ostatni to wynik
                im = ax.imshow(data, cmap=colormap, aspect='auto')
                ax.set_title(title, fontweight='bold', fontsize=12)
            else:
                im = ax.imshow(data, cmap=colormap, aspect='auto', vmin=global_min, vmax=global_max)
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Ukryj niewykorzystane subplot'y
        for i in range(len(individual_data), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Porównanie: pojedyncze pliki vs wynik nałożenia', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generator heatmap z plików PGM')
    parser.add_argument('input', help='Ścieżka do pliku PGM lub katalogu')
    parser.add_argument('--colormap', '-c', default='viridis', 
                       help='Mapa kolorów (domyślnie: viridis)')
    parser.add_argument('--output', '-o', help='Katalog wyjściowy dla zapisanych obrazków')
    parser.add_argument('--compare', action='store_true', 
                       help='Porównaj różne mapy kolorów dla jednego pliku')
    parser.add_argument('--overlay', action='store_true',
                       help='Nałóż wszystkie pliki na siebie i stwórz jedną heatmapę')
    parser.add_argument('--method', default='mean',
                       choices=['mean', 'sum', 'max', 'min', 'median', 'std'],
                       help='Metoda nałożenia plików (domyślnie: mean)')
    parser.add_argument('--show-individual', action='store_true',
                       help='Pokaż także pojedyncze pliki przy nałożeniu')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Nie przeszukuj podkatalogów rekurencyjnie')
    
    args = parser.parse_args()
    
    generator = PGMHeatmapGenerator()
    
    if not os.path.exists(args.input):
        print(f"Błąd: Ścieżka {args.input} nie istnieje")
        return
    
    if args.compare and os.path.isfile(args.input):
        generator.compare_colormaps(args.input)
    elif args.overlay:
        if os.path.isfile(args.input):
            print("Błąd: --overlay wymaga katalogu, nie pojedynczego pliku")
            return
        recursive = not args.no_recursive
        generator.create_overlay_heatmap(args.input, method=args.method, 
                                       colormap=args.colormap, recursive=recursive,
                                       output_dir=args.output, 
                                       show_individual=args.show_individual)
    elif os.path.isfile(args.input):
        generator.process_single_file(args.input, args.colormap, args.output)
    elif os.path.isdir(args.input):
        recursive = not args.no_recursive
        generator.process_directory(args.input, args.colormap, args.output, recursive)
    else:
        print("Błąd: Podana ścieżka nie jest ani plikiem ani katalogiem")

# Przykład użycia bez argumentów wiersza poleceń
if __name__ == "__main__":
    # Sprawdź czy są argumenty wiersza poleceń
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Przykład interaktywnego użycia
        print("=== Generator Heatmap z plików PGM ===")
        print("Przykłady użycia:")
        print("1. python program.py plik.pgm")
        print("2. python program.py katalog/ --colormap hot --output wyniki/")
        print("3. python program.py katalog/ --no-recursive  # bez podkatalogów")
        print("4. python program.py plik.pgm --compare  # porównaj mapy kolorów")
        print("5. python program.py katalog/ --compare-files  # porównaj wszystkie pliki")
        print("6. python program.py katalog/ --compare-files --max-files 9")
        print()
        
        generator = PGMHeatmapGenerator()
        
        # Przykład tworzenia testowych danych
        print("Tworzenie przykładowej heatmapy z losowych danych...")
        test_data = np.random.rand(50, 50) * 255
        generator.create_heatmap(test_data, "Przykładowa heatmapa", colormap='viridis')
        
        # Przykład porównywania plików
        print("Możesz także porównać pliki programowo:")
        print("generator.compare_directory('katalog/', colormap='hot', max_files=9)")
        print("generator.compare_files(['plik1.pgm', 'plik2.pgm'], colormap='plasma')")
        
        print(f"\nObsługiwane mapy kolorów: {', '.join(generator.supported_colormaps)}")