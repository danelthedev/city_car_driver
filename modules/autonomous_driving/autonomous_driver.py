from modules.car_control.car_controller import CarController
from modules.lane_detection.lane_detector import LaneDetector
from modules.traffic_sign_detection.TrafficSignDetector import TrafficSignDetector


class AutonomousDriver:
    def __init__(self, car_controller: CarController, lane_detector: LaneDetector, traffic_sign_detector: TrafficSignDetector):
        self.car_controller = car_controller
        self.lane_detector = lane_detector
        self.traffic_sign_detector = traffic_sign_detector
