import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import pygame
from cv_bridge import CvBridge

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        pygame.init()
        self.screen = None
    
    def image_callback(self, msg):
        # Konwersja obrazu ROS2 na tablicę numpy
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        if self.screen is None:
            self.screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                pygame.quit()

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        pygame.quit()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
