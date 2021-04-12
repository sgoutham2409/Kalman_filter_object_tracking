import numpy as np 
import matplotlib.pyplot as plt 

from kalman_filter import KalmanFilter

plt.ion()
plt.figure()

kf = KalmanFilter(initial_x= 0.4, initial_v= 1.0, acceleration_variance= 0.1)

DT = 0.1
NUM_STEPS = 1000
MEASUREMENT_FREQUENCY = 20 # Assuming we get measurements every 20 steps

measurement_variance = 0.1 ** 2

# True position and velocity
real_position = 0.0 
real_velocity = 0.3

means = []
covariances = []
real_positions = []
real_velocities = []

for step in range(NUM_STEPS):
    if step > 500:
        real_velocity *= 0.9 # Gradually reduce the velocity to 0

    covariances.append(kf.state_cov)
    means.append(kf.state_mean)

    real_position = real_position + real_velocity * DT

    kf.predict(dt= DT)
    if step != 0 and step % MEASUREMENT_FREQUENCY == 0:
        kf.update(measurement_value= real_position + np.random.randn() * np.sqrt(measurement_variance), 
                  measurement_variance= measurement_variance)

    real_positions.append(real_position)
    real_velocities.append(real_velocity)

# plot the real and the estimated positions
plt.subplot(2,1,1)
plt.title('Position')
plt.plot([mu[0] for mu in means], 'r') # mu[0] contains the estimated position in the mean state vector
plt.plot(real_positions, 'b')
# 2 standard deviations bound
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(means, covariances)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(means, covariances)], 'r--')

# plot the real and the estimated velocities
plt.subplot(2,1,2)
plt.title('Velocity')
plt.plot([mu[1] for mu in means], 'r') # mu[1] contains the estimated velocity in the mean state vector
plt.plot(real_velocities, 'b')
# 2 standard deviations bound
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(means, covariances)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(means, covariances)], 'r--')

plt.show()
plt.ginput(1)
