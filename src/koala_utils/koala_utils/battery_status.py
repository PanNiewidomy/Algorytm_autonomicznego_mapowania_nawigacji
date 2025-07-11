import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
import cv2
import numpy as np

class BatteryDisplay(Node):
    def __init__(self):
        super().__init__('battery_display')
        
        # Subskrypcja tematu /battery
        self.subscription = self.create_subscription(
            BatteryState,
            '/battery',
            self.battery_callback,
            10
        )
        
        # Inicjalizacja okna OpenCV
        self.window_name = "Battery Status"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 300, 200)

    def battery_callback(self, msg):
        """Callback dla tematu /battery."""
        # Pobierz poziom naładowania baterii (w procentach)
        battery_percentage = int((msg.voltage-9.6)/3 * 100)
        
        # Wyświetl poziom baterii w oknie OpenCV
        self.display_battery_status(battery_percentage)

    def display_battery_status(self, percentage):
        """Wyświetl stan baterii w oknie OpenCV."""
        # Stwórz czarne tło
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Dodaj tekst z poziomem baterii
        text = f"Battery: {percentage}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        
        # Wyświetl obraz
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def destroy_node(self):
        """Zamknij okno OpenCV przy zamykaniu węzła."""
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    battery_display = BatteryDisplay()
    
    try:
        rclpy.spin(battery_display)
    except KeyboardInterrupt:
        pass
    finally:
        battery_display.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()