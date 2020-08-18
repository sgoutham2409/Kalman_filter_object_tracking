import numpy as np 

# indexes of each variable in the state vector
iPos = 0
iVel = 1

class KalmanFilter:
    def __init__(self, initial_x : float, initial_v : float, acceleration_variance: float) -> None:
        #mean of the state rv, state = [x, v]'
        self._state = np.array([initial_x, initial_v])
        self._acceleration_variance = acceleration_variance

        #covariance matrix of the state rv
        self._P = np.eye(2)

    def predict(self, dt : float) -> None:
        # state = F * state
        # P = F * P * Ft + G * Gt* a
        F = np.array([[1 ,dt], [0, 1]])
        new_state = F.dot(self._state)

        G = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._acceleration_variance

        self._state = new_state
        self._P = new_P
    
    def update(self, measurement_value: float, measurement_variance: float) -> None:
        # y = z - H * state
        # S = H * P * Ht + R
        # K = P * Ht* S ^-1
        # state = state + k * y
        # P = (I - K * H) * P

        H = np.array([1, 0]).reshape((1,2))

        z = np.array([measurement_value])
        R = np.array([measurement_variance])
        
        y = z - H.dot(self._state)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_state = self._state + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P )

        self._state = new_state
        self._P = new_P

    @property
    def pos(self) -> float:
        return self._state[0]
    
    @property
    def vel(self) -> float:
        return self._state[1]

    @property
    def state_mean(self) -> np.array:
        return self._state
    
    @property
    def state_cov(self) -> np.array:
        return self._P