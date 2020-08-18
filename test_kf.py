import numpy as np 

from kalman_filter import KalmanFilter
from unittest import TestCase

class TestKalmanFilter(TestCase):

    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 1.6

        kf = KalmanFilter(initial_x=x, initial_v=v, acceleration_variance=1.3)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_can_predict_and_is_of_right_shape(self):
        x = 0.2
        v = 1.6

        kf = KalmanFilter(initial_x=x, initial_v=v, acceleration_variance=1.3)
        kf.predict(dt=0.1)

        self.assertEqual(kf.state_cov.shape, (2,2))
        self.assertEqual(kf.state_mean.shape, (2,))


    def test_calling_predict_increases_covariance(self):
        x = 0.2
        v = 1.6

        kf = KalmanFilter(initial_x=x, initial_v=v, acceleration_variance=1.3)

        for i in range(10):
            det_before = np.linalg.det(kf.state_cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.state_cov)

            self.assertGreater(det_after, det_before)
            print(det_before, det_after) 
    
    def test_calling_update_does_not_crash(self):
        x = 0.2
        v = 0.3

        kf = KalmanFilter(initial_x= x, initial_v= v, acceleration_variance= 1.3)
        kf.update(measurement_value= 0.1, measurement_variance= 0.1)

    def test_calling_update_decreases_covariance(self):
        x = 0.2
        v = 2.3

        kf = KalmanFilter(initial_x=x, initial_v=v, acceleration_variance=1.2)

        det_before = np.linalg.det(kf.state_cov)
        kf.update(measurement_value= 0.1, measurement_variance= 0.01)
        det_after = np.linalg.det(kf.state_cov)

        self.assertLess(det_after, det_before)

