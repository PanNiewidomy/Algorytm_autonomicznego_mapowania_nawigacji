import sys
import os
import signal
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QFileDialog,
    QCheckBox, QLineEdit, QMessageBox, QListWidgetItem,
    QSizePolicy
)
from PyQt5.QtCore import QTimer, QProcess, Qt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ament_index_python.packages import get_package_share_directory

# for killing gzserver and other processes
import psutil

# For plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MonitoringNode(Node):
    def __init__(self):
        super().__init__('gui_monitor')

    def get_topics(self):
        return self.get_topic_names_and_types()

    def get_services(self):
        return self.get_service_names_and_types()

    def get_nodes(self):
        return self.get_node_names_and_namespaces()

    def destroy_lifecycle_nodes(self):
        """Simplified lifecycle shutdown using subprocess - all nodes in parallel"""
        import subprocess
        import time
        import threading
        
        # Get all lifecycle nodes
        lifecycle_nodes = []
        for node_name, _ in self.get_nodes():
            try:
                services = [name for name, _ in self.get_services()]
                if f'/{node_name}/get_state' in services:
                    lifecycle_nodes.append(node_name)
            except:
                pass
        
        print(f"Found lifecycle nodes: {lifecycle_nodes}")
        
        if not lifecycle_nodes:
            print("No lifecycle nodes found")
            return
        
        def shutdown_node(node):
            """Shutdown a single node"""
            try:
                print(f"Shutting down {node}...")
                
                # Get current state first
                try:
                    result = subprocess.run([
                        'ros2', 'lifecycle', 'get', node
                    ], capture_output=True, text=True, timeout=2)
                    
                    if result.returncode == 0:
                        print(f"{node} state: {result.stdout.strip()}")
                except subprocess.TimeoutExpired:
                    print(f"Timeout getting state of {node}")
                except Exception as e:
                    print(f"Error getting state of {node}: {e}")
                
                # Try to shutdown using ros2 lifecycle command
                try:
                    result = subprocess.run([
                        'ros2', 'lifecycle', 'set', node, 'shutdown'
                    ], capture_output=True, text=True, timeout=10.0)
                    
                    if result.returncode == 0:
                        print(f"✅ {node} shutdown SUCCESS")
                    else:
                        print(f"❌ {node} shutdown FAILED: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ Timeout shutting down {node}")
                except Exception as e:
                    print(f"❌ Error shutting down {node}: {e}")
                    
            except Exception as e:
                print(f"Error processing {node}: {e}")
        
        # Start all shutdowns in parallel using threads
        threads = []
        for node in lifecycle_nodes:
            thread = threading.Thread(target=shutdown_node, args=(node,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete with timeout
        for thread in threads:
            thread.join(timeout=15.0)
        
        print("Lifecycle shutdown completed")

class LidarWindow(QWidget):
    def __init__(self, topic, parent_node):
        super().__init__()
        self.setWindowTitle(f"Lidar Live: {topic}")
        self.resize(1200, 400)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)
        # Cartesian plot
        self.figure_cart = Figure()
        self.canvas_cart = FigureCanvas(self.figure_cart)
        self.canvas_cart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_cart = self.figure_cart.add_subplot(111)
        # Polar scatter
        self.figure_polar = Figure()
        self.canvas_polar = FigureCanvas(self.figure_polar)
        self.canvas_polar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_polar = self.figure_polar.add_subplot(111, projection='polar')
        # Raw data
        self.figure_raw = Figure()
        self.canvas_raw = FigureCanvas(self.figure_raw)
        self.canvas_raw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_raw = self.figure_raw.add_subplot(111)
        # Add canvases
        layout.addWidget(self.canvas_cart, 1)
        layout.addWidget(self.canvas_polar, 1)
        layout.addWidget(self.canvas_raw, 1)
        self.node = parent_node
        self.sub = self.node.create_subscription(LaserScan, topic, self.cb_scan, 10)

    def cb_scan(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        # Cartesian
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        self.ax_cart.clear()
        self.ax_cart.scatter(xs, ys, s=1)
        self.ax_cart.set_aspect('equal')
        self.ax_cart.set_title('Cartesian')
        self.ax_cart.set_xlabel('X')
        self.ax_cart.set_ylabel('Y')
        self.ax_cart.grid()
        # Polar
        self.ax_polar.clear()
        self.ax_polar.scatter(angles, ranges, s=1)
        self.ax_polar.set_title('Polar Scatter')
        finite = ranges[np.isfinite(ranges)]
        if finite.size:
            self.ax_polar.set_ylim(0, np.max(finite) + 0.1)
        # Raw
        self.ax_raw.clear()
        self.ax_raw.scatter(np.arange(len(ranges)), ranges, s=1)
        self.ax_raw.set_title('Raw Data')
        self.ax_raw.grid()
        # Draw
        self.canvas_cart.draw()
        self.canvas_polar.draw()
        self.canvas_raw.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS2 Debug GUI")
        self.resize(900, 700)

        # 1) Stworzenie wszystkich widgetów:
        # Row 1
        self.sim_btn         = QPushButton("Launch Simulation")
        self.gui_box         = QCheckBox("Launch GUI")
        self.rviz_box        = QCheckBox("Launch Rviz")
        self.teleop_btn      = QPushButton("Launch Teleop")
        self.stop_sim_btn    = QPushButton("Stop Simulation")
        # Row 2
        self.use_sim_cb      = QCheckBox("Use Simulation Time")
        self.navi_box        = QCheckBox("Use Navigation")
        self.navi_params_btn = QPushButton("Select Navigation Params")
        self.navi_params_label = QLabel("<default>")
        self.map_btn         = QPushButton("Select Map File")
        self.map_label       = QLabel("<no map selected>")
        # Row 3
        self.global_attempts_label = QLabel("Global Loc Attempts:")
        self.global_attempts = QLineEdit("5")
        self.launch_btn      = QPushButton("Launch Husarion")
        self.stop_hus_btn    = QPushButton("Stop Husarion")
        # Row 4
        self.expl_rviz_box   = QCheckBox("Launch Explore RViz")
        self.expl_rviz_label = QLabel("<rviz config>")
        self.expl_params_btn = QPushButton("Select Explore Params")
        self.expl_params_label = QLabel("<default>")
        self.launch_expl_btn = QPushButton("Launch Exploration")
        self.stop_expl_btn   = QPushButton("Stop Exploration")
        # … i reszta widgetów …

        # 2) Teraz nadajemy objectName:
        for btn in (self.sim_btn, self.launch_btn, self.launch_expl_btn):
            btn.setObjectName("btnStart")
        for btn in (self.stop_sim_btn, self.stop_hus_btn, self.stop_expl_btn):
            btn.setObjectName("btnStop")
        self.rviz_box.setObjectName("rvizConfig")
        self.expl_rviz_box.setObjectName("rvizConfig")
        self.expl_rviz_label.setObjectName("rvizConfigLabel")


        central = QWidget()
        layout = QVBoxLayout(central)

        # Row 1: Simulation
        row1 = QHBoxLayout(); row1.setSpacing(10)
        for w in (self.gui_box, self.rviz_box, self.teleop_btn, self.sim_btn, self.stop_sim_btn):
            row1.addWidget(w)
        layout.addLayout(row1)

        # Row 2: Navigation/Husarion params
        row2 = QHBoxLayout(); row2.setSpacing(10)
        for w in (self.use_sim_cb, self.navi_box, self.map_label, self.map_btn, self.navi_params_label, self.navi_params_btn):
            row2.addWidget(w)
        layout.addLayout(row2)

        # Row 3: Husarion launch
        row3 = QHBoxLayout(); row3.setSpacing(10)
        for w in (self.global_attempts_label,
                  self.global_attempts, self.launch_btn, self.stop_hus_btn):
            row3.addWidget(w)
        layout.addLayout(row3)

        # Row 4: Exploration controls
        row4 = QHBoxLayout(); row4.setSpacing(10)
        for w in (self.expl_rviz_box, self.expl_rviz_label,
                self.expl_params_label, self.expl_params_btn,
                self.launch_expl_btn, self.stop_expl_btn):
            row4.addWidget(w)
        layout.addLayout(row4)

        # 3) Nadajesz objectName i style **po** pełnym zbudowaniu UI
        for btn in (self.sim_btn, self.launch_btn, self.launch_expl_btn):
            btn.setObjectName("btnStart")
        for btn in (self.stop_sim_btn, self.stop_hus_btn, self.stop_expl_btn):
            btn.setObjectName("btnStop")
        self.rviz_box.setObjectName("rvizConfig")
        self.expl_rviz_box.setObjectName("rvizConfig")
        self.expl_rviz_label.setObjectName("rvizConfigLabel")

        self.setStyleSheet("""
            QWidget { font-family: Arial; font-size: 14px; }
            QPushButton { border-radius: 8px; padding: 8px; background-color: #3A7BD5; color: white; }
            QPushButton:hover { background-color: #2A5DA8; }
            QPushButton#btnStart { background-color: #4CAF50; }
            QPushButton#btnStart:hover { background-color: #45A049; }
            QPushButton#btnStop  { background-color: #F44336; color: #000000; }
            QPushButton#btnStop:hover  { background-color: #D32F2F; }
            QCheckBox#rvizConfig, QLabel#rvizConfigLabel { color: #632d02; }
            QLineEdit, QLabel { padding: 4px; }
            QListWidget { border: 1px solid #ccc; }
        """)

        self.setCentralWidget(central)

        # Default exploration params and rviz config paths
        try:
            pkg_navi = get_package_share_directory('koala_navigation')
            default_expl = os.path.join(pkg_navi, 'params', 'nav2_params_final.yaml')
        except:
            default_expl = ''
        self.navi_params_file = default_expl
        self.navi_params_label.setText(os.path.basename(default_expl) if default_expl else "<no params>")

        # Default exploration params and rviz config paths
        try:
            pkg_expl = get_package_share_directory('koala_exploration')
            default_expl = os.path.join(pkg_expl, 'params', 'explore.yaml')
            default_rviz = os.path.join(pkg_expl, 'params', 'explore.rviz')
        except:
            default_expl = ''
            default_rviz = ''
        self.expl_params_file = default_expl
        self.expl_params_label.setText(os.path.basename(default_expl) if default_expl else "<no params>")
        self.expl_rviz_file = default_rviz
        self.expl_rviz_label.setText(os.path.basename(default_rviz) if default_rviz else "<no rviz>")

        # Lists of topics/services/nodes
        lists = QHBoxLayout()
        self.topics_list = QListWidget()
        self.services_list = QListWidget()
        self.lifecycle_list = QListWidget()
        for w,title in ((self.topics_list,"Topics"),
                        (self.services_list,"Services"),
                        (self.lifecycle_list,"Lifecycle Nodes")):
            v = QVBoxLayout()
            v.addWidget(QLabel(title)); v.addWidget(w)
            lists.addLayout(v)
        layout.addLayout(lists)

        # Publish/Call
        action = QHBoxLayout()
        self.pub_input = QLineEdit()
        self.pub_btn = QPushButton("Publish/Call")
        action.addWidget(self.pub_input); action.addWidget(self.pub_btn)
        layout.addLayout(action)

        # Process lists
        self.child_procs_sim = []
        self.child_procs_navi = []
        self.child_procs_expl = []
        self.child_procs_teleop = []

        # Init files 
        self.map_file = ""
        self.expl_params_file = ""
        self.navi_params_file = ""

        # Signal connections
        self.sim_btn.clicked.connect(self.launch_sim)
        self.stop_sim_btn.clicked.connect(lambda: self.stop_nodes(self.child_procs_sim, "Simulation"))
        self.launch_btn.clicked.connect(self.launch_husarion)
        self.stop_hus_btn.clicked.connect(lambda: self.stop_nodes(self.child_procs_navi, "Navigation"))
        self.launch_expl_btn.clicked.connect(self.launch_exploration)
        self.stop_expl_btn.clicked.connect(lambda: self.stop_nodes(self.child_procs_expl, "Exploration"))

        self.teleop_btn.clicked.connect(self.run_teleop) 

        self.map_btn.clicked.connect(lambda: self.select_params('map_file', self.map_label, "map"))
        self.expl_params_btn.clicked.connect(lambda: self.select_params('expl_params_file', self.expl_params_label, "exploration"))
        self.navi_params_btn.clicked.connect(lambda: self.select_params('navi_params_file', self.navi_params_label, "navigation"))

        self.use_sim_cb.stateChanged.connect(self.update_default_map)
        self.topics_list.itemDoubleClicked.connect(self.handle_topic)
        self.services_list.itemDoubleClicked.connect(self.handle_service)
        self.lifecycle_list.itemDoubleClicked.connect(self.handle_lifecycle)
        self.topics_list.currentItemChanged.connect(self.prepare_action)
        self.services_list.currentItemChanged.connect(self.prepare_action)
        self.pub_btn.clicked.connect(self.execute_action)


        # ROS2 init & timer
        rclpy.init()
        self.node = MonitoringNode()
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)

    def select_params(self, _file, _label, _name):
        fname, _ = QFileDialog.getOpenFileName(
            self, f"Select {_name} file", "", "YAML Files (*.yaml)"
        )
        if fname:
            setattr(self, _file, fname)
            _label.setText(os.path.basename(fname))

    def run_teleop(self):
        proc = QProcess(self)
        proc.start(
            'xterm', ['-T', 'teleop', '-hold', '-e', 'ros2', 'run', 'teleop_twist_keyboard', 'teleop_twist_keyboard']
        )
        self.child_procs_navi.append(proc)


    def launch_sim(self):
        if self.child_procs_sim:
            self.stop_nodes(self.child_procs_sim, "Simulation")
        try:
            pkg = get_package_share_directory('koala_simulation')
            launch_path = os.path.join(pkg, 'launch', 'gazebo.house.launch.py')
        except:
            launch_path = 'gazebo.house.launch.py'
        gui = str(self.gui_box.isChecked()).lower()
        rviz = str(self.rviz_box.isChecked()).lower()
        cmd = ['ros2','launch', launch_path,
               f'gui:={gui}', f'rviz_run:={rviz}']
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.child_procs_sim.append(p)

    def launch_husarion(self):
        print(self.navi_params_file)
        if self.child_procs_navi:
            self.stop_nodes(self.child_procs_navi, "Navigation")
        try:
            pkg = get_package_share_directory('koala_navigation')
            launch_path = os.path.join(pkg, 'launch', 'husarion_launch.py')
        except:
            launch_path = 'husarion_launch.py'
        sim = str(self.use_sim_cb.isChecked()).lower()
        navi = str(self.navi_box.isChecked()).lower()
        cmd = ['ros2','launch', launch_path,
               f'params_file:={self.navi_params_file}',
               f'use_sim_time:={sim}', f'map:={self.map_file}',
               f'navigation_on:={navi}',
               f'num_attempts:={int(self.global_attempts.text())}']
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.child_procs_navi.append(p)

    def launch_exploration(self):
        if self.child_procs_expl:
            self.stop_nodes(self.child_procs_expl, "Exploration")
        try:
            pkg = get_package_share_directory('koala_exploration')
            launch_path = os.path.join(pkg, 'launch', 'exploration_node_launch.py')
            if self.expl_rviz_box.isChecked():
                print(self.expl_rviz_file)
                p=subprocess.Popen(['ros2','run','rviz2','rviz2', '-d', self.expl_rviz_file], preexec_fn=os.setsid)
                self.child_procs_expl.append(p)
        except:
            launch_path = 'exploration_node_launch.py'
        sim = str(self.use_sim_cb.isChecked()).lower()
        params = self.expl_params_file
        cmd = ['ros2','launch', launch_path,
               f'use_sim_time:={sim}', f'params_file:={params}']
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.child_procs_expl.append(p)

    def stop_nodes(self, processes_list, node_name):
        self.node.destroy_lifecycle_nodes()
        if processes_list:
            for p in processes_list:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(1)
                except:
                    pass
            processes_list.clear()
            if node_name == "Simulation":
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] == 'gzserver':
                        try: proc.kill()
                        except: pass
            QMessageBox.information(self, "Info", f"{node_name} launch stopped.")
        else:
            QMessageBox.warning(self, "Error", f"{node_name} is not running.")
  

    def update_default_map(self):
        try:
            pkg = get_package_share_directory('koala_navigation')
            rel = ['maps', 'Symulacja', 'Szum', 'map.yaml'] if self.use_sim_cb.isChecked() else ['maps', 'Sala', 'map.yaml']
            default = os.path.join(pkg, *rel)
            if os.path.exists(default):
                self.map_file = default
                self.map_label.setText(os.path.basename(default))
            else:
                raise FileNotFoundError
        except:
            self.map_file = ''
            self.map_label.setText("<no map selected>")

    def refresh(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        
        # Update topics list
        self.topics_list.clear()
        for name, types in self.node.get_topics():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, (name, types[0] if types else None))
            self.topics_list.addItem(item)
        
        # Update services list
        self.services_list.clear()
        for name, types in self.node.get_services():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, (name, types[0] if types else None))
            self.services_list.addItem(item)
        
        # Update lifecycle nodes list
        self.lifecycle_list.clear()
        lifecycle_nodes = []
        for node_name, _ in self.node.get_nodes():
            try:
                services = [sname for sname, _ in self.node.get_services()]
                if f'/{node_name}/get_state' in services:
                    lifecycle_nodes.append(node_name)
            except:
                pass
        
        for node_name in lifecycle_nodes:
            item = QListWidgetItem(node_name)
            item.setData(Qt.ItemDataRole.UserRole, (node_name, None))
            self.lifecycle_list.addItem(item)

    def handle_topic(self, item):
        name, ttype = item.data(Qt.ItemDataRole.UserRole)
        proc = QProcess(self)
        window_title = f"Topic: {name}"
        proc.start(
            'xterm', ['-T', window_title, '-hold', '-e', 'ros2', 'topic', 'echo', name]
        )
        self.child_procs_navi.append(proc)
        # Launch battery status script on /battery click
        if name == '/battery':
            try:
                pkg_utils = get_package_share_directory('koala_utils')
                battery_script = os.path.join(pkg_utils, 'battery_status.py')
                subprocess.Popen(['python3', battery_script], preexec_fn=os.setsid)
            except Exception as e:
                QMessageBox.warning(
                    self, 'Error', f"Could not start battery_status.py:\n{e}"
                )
        # Lidar window for LaserScan
        if ttype == 'sensor_msgs/msg/LaserScan':
            self.lidar_win = LidarWindow(name, self.node)
            self.lidar_win.show()

    def handle_service(self, item):
        name, stype = item.data(Qt.ItemDataRole.UserRole)
        proc = QProcess(self)
        window_title = f"Service: {name}"
        proc.start(
            'xterm', ['-T', window_title, '-hold', '-e', 'ros2', 'service', 'type', name]
        )
        self.child_procs_navi.append(proc)

    def handle_lifecycle(self, item):
        name, _ = item.data(Qt.ItemDataRole.UserRole)
        proc = QProcess(self)
        window_title = f"Lifecycle: {name}"
        proc.start(
            'xterm', ['-T', window_title, '-hold', '-e', 'ros2', 'lifecycle', 'get', name]
        )
        self.child_procs_navi.append(proc)

    def prepare_action(self, current, previous):
        if not current:
            return
        name, ttype = current.data(Qt.ItemDataRole.UserRole)
        sender = self.sender()
        self.pub_target = 'topic' if sender == self.topics_list else 'service'
        self.pub_name = name
        self.pub_type = ttype
        if ttype and 'String' in ttype:
            default = "{'data':'Hello'}"
        else:
            default = '{}'
        self.pub_input.setText(default)
        self.pub_btn.setText('Publish' if self.pub_target=='topic' else 'Call')

    def execute_action(self):
        if not hasattr(self, 'pub_name') or not self.pub_name or not hasattr(self, 'pub_type') or not self.pub_type:
            QMessageBox.warning(self, 'Error', 'Select a topic or service first')
            return
        msg = self.pub_input.text().strip()
        if self.pub_target == 'topic':
            cmd = ['ros2','topic','pub','--once', self.pub_name, self.pub_type, msg]
        else:
            cmd = ['ros2','service','call', self.pub_name, self.pub_type, msg]
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.child_procs_navi.append(p)

def main(args=None):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
