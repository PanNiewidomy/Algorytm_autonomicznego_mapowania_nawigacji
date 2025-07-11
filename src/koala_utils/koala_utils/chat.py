import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_pgm_files(root_dir):
    pgm_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.pgm'):
                pgm_files.append(os.path.join(dirpath, filename))
    return pgm_files

def load_and_resize_pgm(path, target_size):
    img = Image.open(path)
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img, dtype=np.float32)

def main(root_dir, heatmap_output="heatmap.png"):
    pgm_files = find_pgm_files(root_dir)
    if not pgm_files:
        print("Nie znaleziono żadnych plików .pgm.")
        return

    # Ustalamy rozmiar odniesienia (pierwszy obraz)
    with Image.open(pgm_files[0]) as img0:
        target_size = img0.size  # (width, height)

    sum_image = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
    count = 0

    for pgm_file in pgm_files:
        img = load_and_resize_pgm(pgm_file, target_size)
        sum_image += img
        count += 1

    avg_image = sum_image / count

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(avg_image, cmap='hot')
    plt.tight_layout()
    plt.savefig(heatmap_output, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Heatmapa zapisana jako {heatmap_output}")
    
if __name__ == "__main__":
    main("/home/jakub/dev_magisterka/Pomiary/Symulacja")  # <- wpisz swoją ścieżkę
    # 