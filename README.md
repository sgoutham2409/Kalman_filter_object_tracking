# Kalman_filter_object_tracking
This project is an attempt at understanding the working of Kalman filter for estimating the unknown state of a system based on a series of measurements and the prior knowledge of the system motion model. 
Here the state of an object, [position, velocity]', is estimated with a higher accuracy by combining the less accurate measurement of the object's state based on detection (for instance, it could be background subtraction) and the less accurate predictions of the object's state based on the prior motion model.
 
Kalman filter is implemented in Python with the following assumptions:
  1. A linear motion model, noisy acceleration model
  2. Measurement corrupted by a gaussian noise

Unittest framework and Pytest used to create and run simple tests to validate the Kalman filter behavior

My takeaway: The key idea of Kalman filter is to combine multiple sources of inaccurate data representing the real value to get a more accurate estimate for the real value than each of the independent data sources.
