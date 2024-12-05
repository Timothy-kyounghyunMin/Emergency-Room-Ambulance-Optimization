import numpy as np
import random
from scipy.stats import expon
from collections import deque
from math import radians, cos, sin, asin, sqrt, log, tan, atan2, degrees, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import imageio
from tqdm import tqdm
import numpy as np
from collections import deque

IMAGE_DIR = "simulation_images"  # Directory to store images
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Constants
WIDTH = (126.7642937, 127.1837947)  # Approximate longitude range for Seoul
HEIGHT = (37.4259627, 37.7017496)  # Approximate latitude range for Seoul
VELOCITY = 30  # ER van velocity (km/h), approx 8.33 m/s
UNIT_TIME = 1  # Set the unit time `u` as 1 second for simulation
LAMBDA = 0.5  # The lambda for exponential distribution, placeholder value
TREATMENT_LAMBDA = 0.1  # The lambda for exponential distribution, placeholder value
MAX_SIMULATION_TIME = 3600  # 1 hour of simulation time for example
MAX_WAIT_TIME_FOR_TREATMENT = 15 * 60  # Max wait time for treatment in seconds


# Data structures
# Data structures
hospitals = {
    'Seoul National University Hospital': {'location': (126.9991, 37.5796), 'queue': deque(), 'current_patient': None},
    'Asan Medical Center': {'location': (127.1079, 37.5262), 'queue': deque(), 'current_patient': None},
    'Samsung Medical Center': {'location': (127.0853, 37.4881), 'queue': deque(), 'current_patient': None},
    'Severance Hospital': {'location': (126.9408, 37.5622), 'queue': deque(), 'current_patient': None},
    'Korea University Guro Hospital': {'location': (126.8849, 37.4914), 'queue': deque(), 'current_patient': None},
    # ... Add additional hospitals as needed
}
patients = []
simulation_time = 0
DEATH_TOLL = 0
TREATED_PATIENTS = 0
TOTAL_ETAW = 0


# Define the Patient class
class Patient:
    def __init__(self, location, arrival_time, status='waiting'):
        self.location = location
        self.arrival_time = arrival_time
        self.status = status
        self.assigned_hospital = None
        self.treatment_time = None

    def assign_hospital(self, hospital):
        self.assigned_hospital = hospital

    def transport(self):
        self.status = 'transported'

    def start_treatment(self, treatment_time):
        self.status = 'in_treatment'
        self.treatment_time = treatment_time

    def update_location(self, new_location):
        self.location = new_location

    def update_treatment_time(self, elapsed_time):
        if self.treatment_time is not None:
            self.treatment_time -= elapsed_time
            if self.treatment_time <= 0:
                self.finish_treatment()

    def finish_treatment(self):
        self.status = 'treated'
        self.treatment_time = None

    def is_treatment_finished(self):
        return self.treatment_time is not None and self.treatment_time <= 0



def generate_patient_location():
    longitude = random.uniform(*WIDTH)
    latitude = random.uniform(*HEIGHT)
    return (longitude, latitude)

def calculate_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    
    :param pointA: Tuple for the starting point (longitude, latitude).
    :param pointB: Tuple for the end point (longitude, latitude).
    
    :return: Bearing as a float in degrees.
    """
    startLat = radians(pointA[1])
    startLong = radians(pointA[0])
    endLat = radians(pointB[1])
    endLong = radians(pointB[0])
    
    dLong = endLong - startLong
    
    dPhi = log(tan(endLat / 2.0 + pi / 4.0) / tan(startLat / 2.0 + pi / 4.0))
    if abs(dLong) > pi:
        if dLong > 0.0:
            dLong = -(2.0 * pi - dLong)
        else:
            dLong = (2.0 * pi + dLong)
    
    bearing = (degrees(atan2(dLong, dPhi)) + 360.0) % 360.0
    
    return bearing

def sample_exponential_time(lambda_value):
    return expon.rvs(scale=1/lambda_value)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000  # Return distance in meters

def calculate_new_position(lat, lon, bearing, distance):
    # Earth radius in meters
    R = 6371e3

    # Convert latitude and longitude from degrees to radians
    lat_rad = radians(lat)
    lon_rad = radians(lon)

    # Convert bearing to radians
    bearing_rad = radians(bearing)

    # Calculate the new latitude
    new_lat_rad = asin(sin(lat_rad) * cos(distance / R) +
                       cos(lat_rad) * sin(distance / R) * cos(bearing_rad))

    # Calculate the new longitude
    new_lon_rad = lon_rad + atan2(sin(bearing_rad) * sin(distance / R) * cos(lat_rad),
                                  cos(distance / R) - sin(lat_rad) * sin(new_lat_rad))

    # Convert the new latitude and longitude back to degrees
    new_lat = degrees(new_lat_rad)
    new_lon = degrees(new_lon_rad)

    return new_lat, new_lon

def update_patient_locations():
    for patient in patients:
        if patient['status'] == 'transported':
            # Calculate the distance the patient has moved
            distance_moved = VELOCITY * 1000 / 3600 * UNIT_TIME  # VELOCITY in m/s * UNIT_TIME in seconds
            # Calculate the bearing to the hospital
            hospital = hospitals[patient['assigned_hospital']]
            bearing = calculate_bearing(patient['location'], hospital['location'])
            # Update the patient's location
            patient['location'] = calculate_new_position(patient['location'][0], patient['location'][1], bearing, distance_moved)
            # Check if the patient has arrived at the hospital
            distance_to_hospital = calculate_distance(patient['location'], hospital['location'])
            if distance_to_hospital < VELOCITY * 1000 / 3600 * UNIT_TIME:  # If the patient has arrived within one time unit
                patient['status'] = 'waiting'  # Change status to 'waiting'
                hospital['queue'].append(patient)  # Add patient to the hospital queue


def plot_simulation(simulation_time, patients, hospitals, IMAGE_DIR):
    fig, ax = plt.subplots()
    ax.set_xlim(WIDTH)
    ax.set_ylim(HEIGHT)
    ax.set_title(f'Simulation Time: {simulation_time} seconds')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    for hospital_id, hospital in hospitals.items():
        x, y = hospital['location']
        ax.plot(x, y, 'ro')  # Red 'o' for hospitals
        ax.text(x, y, hospital_id, fontsize=9)

    for patient in patients:
        if patient['status'] == 'transported':
            x, y = patient['location']
            ax.plot(x, y, 'bo')  # Blue 'o' for patients traveling to hospitals

    plt.savefig(f"{IMAGE_DIR}/frame_{simulation_time:04d}.png")  # Save the frame
    plt.close(fig)  # Close the plot to avoid memory issues


def create_gif():
    images = []
    for file_name in sorted(os.listdir(IMAGE_DIR)):
        if file_name.endswith('.png'):
            file_path = os.path.join(IMAGE_DIR, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('simulation.gif', images, fps=60)  # fps controls the speed


# Replace the simple Euclidean distance calculation with the Haversine distance
def calculate_distance(patient_location, hospital_location):
    return haversine(patient_location[0], patient_location[1], hospital_location[0], hospital_location[1])

def calculate_etaw(distance, queue_length, lambda_value):
    travel_time = distance / (VELOCITY * 1000 / 3600)  # Correct velocity conversion to m/s
    waiting_time = lambda_value * (queue_length + 1 + np.sqrt(queue_length + 1))
    return travel_time + waiting_time

def determine_hospital(patient_location):
    etaws = {
        hospital_id: calculate_etaw(
            calculate_distance(patient_location, hospital['location']),
            len(hospital['queue']),
            LAMBDA
        ) for hospital_id, hospital in hospitals.items()
    }
    return min(etaws, key=etaws.get)

# This function would be part of the simulation loop where new patients arrive
def new_patient_arrival(location, arrival_time):
    # Here you would decide which hospital the patient goes to based on your criteria (e.g., proximity)
    selected_hospital_id = determine_hospital(location)
    new_patient = Patient(location, arrival_time)
    hospitals[selected_hospital_id]['queue'].append(new_patient)
    new_patient.assign_hospital(selected_hospital_id)

# Function to simulate the movement and treatment of patients
def update_hospital_queues():
    global TREATED_PATIENTS, DEATH_TOLL, TOTAL_ETAW
    # Move patients through queues
    for hospital_id, hospital in hospitals.items():
        if hospital['current_patient']:
            # Reduce treatment time as time progresses
            hospital['current_patient'].update_treatment_time(UNIT_TIME)
            if hospital['current_patient'].is_treatment_finished():
                # The patient's treatment has finished
                TREATED_PATIENTS += 1
                hospital['current_patient'] = None  # The patient leaves the hospital
        else:
            # Start treatment if there's a patient waiting and no current patient is being treated
            while hospital['queue'] and hospital['current_patient'] is None:
                next_patient = hospital['queue'].popleft()
                if next_patient.status == 'waiting':  # Make sure to only start treating waiting patients
                    hospital['current_patient'] = next_patient
                    treatment_time = sample_exponential_time(TREATMENT_LAMBDA)
                    next_patient.start_treatment(treatment_time)
                    break  # Found a patient to treat, exit the loop



# Modify the run_simulation function
def run_simulation():
    global simulation_time, TOTAL_ETAW, DEATH_TOLL, TREATED_PATIENTS

    # Sample time intervals from exponential distribution
    time_intervals = np.random.exponential(scale=1/LAMBDA, size=3000)

    # Accumulate the time intervals to get the generation times for each patient
    patient_generation_times = np.cumsum(time_intervals)

    # Convert the generation times to integers
    patient_generation_times = set(patient_generation_times.astype(int))

    print(f"Number of patients: {len(patient_generation_times)}")

    pbar = tqdm(total=MAX_SIMULATION_TIME)  # For visual feedback in notebooks
    
    while simulation_time < MAX_SIMULATION_TIME:
        # Simulate patient occurrence with correct time interval
        if simulation_time in patient_generation_times:
            patient_location = generate_patient_location()
            arrival_time = simulation_time
            new_patient = Patient(patient_location, arrival_time)
            patients.append(new_patient)
            # Assign the best hospital immediately
            best_hospital_id = determine_hospital(new_patient.location)
            hospitals[best_hospital_id]['queue'].append(new_patient)
            new_patient.assign_hospital(best_hospital_id)
            new_patient.transport()

        # Update hospital queues and treat patients
        update_hospital_queues()

        # Check for patients who have been waiting too long and update death toll
        for patient in patients:
            if patient.status == 'waiting':
                patient.time_waited_for_treatment += UNIT_TIME
                if patient.time_waited_for_treatment > MAX_WAIT_TIME_FOR_TREATMENT:
                    patient.status = 'deceased'
                    DEATH_TOLL += 1

        # Update the simulation clock
        simulation_time += UNIT_TIME
        pbar.update(UNIT_TIME)  # Update the progress bar
    
    pbar.close()
    print(f"Death toll: {DEATH_TOLL}")
    print(f"Treated patients: {TREATED_PATIENTS}")
    print(f"Average ETA-waiting: {TOTAL_ETAW / TREATED_PATIENTS if TREATED_PATIENTS > 0 else 'Undefined'}")

    create_gif()  # Uncomment if the create_gif function is implemented


# Now you can run the simulation
run_simulation()