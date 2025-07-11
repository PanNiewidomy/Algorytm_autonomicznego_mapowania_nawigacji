### 1. Konfiguracja środowiska

#### 1.1 Instalacja ROS 2 Humble

1. Dodaj repozytorium ROS 2:

   ```bash
   sudo apt update
   sudo apt install -y software-properties-common curl gnupg2 lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
   sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
   sudo apt update
   ```
2. Zainstaluj środowisko desktopowe ROS 2 Humble:

   ```bash
   sudo apt install -y ros-humble-desktop
   ```

> **Wskazówka:** W razie potrzeby zmień `humble` na wersję odpowiadającą Twojej dystrybucji ROS 2.

#### 1.2 Narzędzia developerskie

```bash
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  build-essential \
  git
```

#### 1.3 Inicjalizacja i aktualizacja rosdep

```bash
sudo rosdep init    # wykonaj raz przy pierwszym użyciu
rosdep update       # aktualizuje bazę zależności
```

---

### 2. Struktura workspace

1. Utwórz strukturę katalogów workspace:

   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

2. Sklonuj repozytorium pakietu do katalogu `src`:

   ```bash
   cd src
   git clone https://github.com/PanNiewidomy/Algorytm_autonomicznego_mapowania_nawigacji.git
   ```

3. **(Opcjonalnie)** Dodaj dodatkowy świat symulacyjny ROS 2 od AWS RoboMaker:

   ```bash
   cd src
   git clone --branch ros2 https://github.com/aws-robotics/aws-robomaker-small-house-world.git
   ```

4. Wróć do katalogu głównego workspace:

   ```bash
   cd ~/ros2_ws
   ```

   ```bash
   cd src
   git clone https://github.com/PanNiewidomy/Algorytm_autonomicznego_mapowania_nawigacji.git
   cd ..
   ```

---

### 3. Instalacja zależności projektu

Wykorzystaj `rosdep`, by automatycznie pobrać i zainstalować wszystkie wymagane biblioteki systemowe i pakiety ROS:

```bash
rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro humble \
  -y
```

* `--from-paths src` – analiza deklaracji zależności w plikach `package.xml` wszystkich pakietów w `src`.
* `--ignore-src` – pominięcie pakietów dostępnych w wersji źródłowej.
* `--rosdistro` – wersja dystrybucji ROS (tu: `humble`).
* `-y` – automatyczna akceptacja instalacji.

---

### 4. Kompilacja pakietu

#### 4.1 Budowanie z wykorzystaniem colcon

1. Przejdź do katalogu głównego workspace:

   ```bash
   cd ~/ros2_ws
   ```
2. Uruchom kompilację:

   ```bash
   colcon build --symlink-install
   ```

   Flaga `--symlink-install` powoduje tworzenie dowiązań symbolicznych do plików źródłowych, co przyspiesza testowanie i rozwój.

#### 4.2 Konfiguracja środowiska po budowaniu

Po zakończeniu procesu kompilacji załaduj zmienne środowiskowe workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

Możesz dodać powyższą linię do pliku `~/.bashrc`, aby automatycznie źródłować środowisko w nowych terminalach.

---

### 5. Uruchamianie i walidacja

1. **Weryfikacja instalacji pakietu**

   ```bash
   ros2 pkg list | grep Algorytm_autonomicznego_mapowania_nawigacji
   ```

2. **Start węzła głównego**
   Zakładając, że w repozytorium znajduje się plik wykonywalny `main_node`:

   ```bash
   ros2 run Algorytm_autonomicznego_mapowania_nawigacji main_node
   ```

3. **Monitorowanie komunikacji**

   * Lista topików:

     ```bash
     ros2 topic list
     ```
   * Obserwacja danych na wybranym topiku (np. `/map_topic`):

     ```bash
     ros2 topic echo /map_topic
     ```

---

### 6. Diagnostyka i optymalizacja

* **Błędy kompilacji**

  * Sprawdź sekcję `find_package(...)` w `CMakeLists.txt` oraz deklaracje `<depend>` w `package.xml`. Upewnij się, że biblioteki wymaganych pakietów są zainstalowane.
  * W razie niezgodności wersji ROS 2 lub bibliotek systemowych skonsultuj dokumentację odpowiednich pakietów.

* **Problemy z rosdep**

  * Zweryfikuj poprawność nazw zależności w `package.xml` względem dostępnych pakietów systemowych.
  * Upewnij się, że masz aktualną bazę `rosdep update`.

* **Częste operacje developerskie**

  * Oczyszczenie pamięci podręcznej CMake i ponowna kompilacja:

    ```bash
    colcon build --symlink-install --cmake-clean-cache
    ```
  * Szybkie ponowne uruchomienie procesu budowania po drobnych zmianach:

    ```bash
    colcon build --packages-select Algorytm_autonomicznego_mapowania_nawigacji --symlink-install
    ```

---

Po wykonaniu powyższych kroków środowisko powinno być przygotowane, a pakiet gotowy do dalszych eksperymentów i rozwoju zaawansowanych algorytmów autonomicznej nawigacji. Powodzenia w pracy nad projektem!
