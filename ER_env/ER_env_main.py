#########################################################
# Import Several Packages
#########################################################
from math import *
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation
import seaborn as sns
import warnings
import json
import sys
import queue
import os

warnings.filterwarnings(action='ignore')

# Get the seed_num through sys
seed_num = int(sys.argv[1])
model_type = sys.argv[2]

# Set Random Seed
np.random.seed(seed_num)

# Create the log file 
with open(f'./log/log_{model_type}_{seed_num}.txt', 'w') as f:
    f.write(f"seed_num: {seed_num}\n")
    f.write(f"model_type: {model_type}\n")
    f.write(f"np.random.seed: {seed_num}\n")


#########################################################
# 구현해야 하는 기본 모듈: map / patient / hospital

#########################################################
######################## For MAP ########################
#########################################################
"""
[ Attributes] 

1. GRID:
	- min_lat								(float)
    - max_lat								(float)
    - min_lon								(float)
    - max_lon								(float)
2. districts: 								(dict)
	{0:[center_location, radius], ...}
    - 0: 관악구
    - center_location: mu_1, mu_2 			(tuple)
    - radius: np.sqrt(size/np.pi) 			(float)
3. weights: 								(list)
	[관악구 인구 %, 그 외 다른 거 %, ]
4. patient_dict:							(dict)
	{id: class Patient(), ...}
5. acc_patient_num:							(int)

[ Methods ]

1. create_patient(self, no arg)				(func)
	- id = self.acc_patient_num
    - d_id = np.random.choice(self.weights)
      lat, lon = np.random.multivariatenormal(self.districts[d_id][0], self.districts[d_id][0])
      lat = max(min(lat, max_lat), min_lat)
      lon = max(min(lon, max_lon), min_lon)
      location = (lat, lon)
    - created_time = tau
    - state = 0
    - destiny = np.random."oracle"
      (oracle: 처음에는 사망 확률이 낮다가, 시간 지나면 상승, left skewed)			[GPT 부탁해!]
    
    (그래서)
    self.patient_dict[id] = Patient(id, location, created_time, state, destiny)
    
    return
"""

# Define Classes: Patient, Hospital

class Patient():
  def __init__(self, id:int, location:tuple, created_time:int, destiny:float):
    self.id = id
    self.location = location   #tuple (float, float)
    self.created_time = created_time
    self.destiny = destiny # 사망 시간
    self.decision = None # Optimizer에서 결정되어 할당 (향하고 있는 병원)
    self.treatment_time = None # Optimizer에서 결정되어 할당 (치료 시간)

    
class Hospital():
  def __init__(self, hospitals_id:int, hospitals_location:tuple, mean_treat_time:int, seed_num:int):
    self.queue = [] # queue.Queue()
    self.len_queue = 0
    self.treat_start = None
    self.id = hospitals_id
    self.location = hospitals_location
    self.mean_treat_time = mean_treat_time
    self.seed = seed_num + self.id
    self.random_state = np.random.RandomState(self.seed)  ########## 12.08 수정 random_state 
  
  def sample_treat_time(self):
    # np.random.seed(seed_num + self.id) ###################### 수정했음 (seed_num + self.id)
    #treat_time = np.random.exponential(self.mean_treat_time, 1)
    treat_time = self.random_state.exponential(self.mean_treat_time, 1)  ########## 12.08 수정 random_state 
    return treat_time
  

def create_patient(model_type: str, patient_location, patient_district, patient_destiny):
    """
    model_type: 'naive' or 'opt'
    """
    global tau, acc_patient_num, weights, districts, GRID 
    
    id = acc_patient_num
    created_time = tau # Global time step

    # Save the string into log file
    with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
        f.write(f"환자 {id}가 생성되었습니다. 위치: {patient_location}, 생성 시간: {created_time}, 사망 시간: {patient_destiny}\n")

    # Assuming Patient class is defined elsewhere
    patient_dict[id] = Patient(id, patient_location, created_time, patient_destiny)
    
    acc_patient_num += 1

    # Determine the hospital that the patient should go using optimizer
    if model_type == 'opt':
      patient_dict[id].decision, _ = optimizer(id)
    elif model_type == 'naive':
      patient_dict[id].decision = naive_optimizer(id)
    else:
      raise ValueError("model_type should be either 'opt' or 'naive'")
    
    ##test
    temp_eta, temp_etaw = compute_etaw(patient_dict[id], patient_dict[id].decision, tau)
    with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
        f.write(f"환자 {id}의 eta: {temp_eta}, etaw: {temp_etaw}\n")

def compute_etaw(patient: Patient, hospital: Hospital, tau: int):
  patient_lon, patient_lat = patient.location
  hospital_lon, hospital_lat = hospital.location

  # Calculate the real distance using l2 norm
  distance = np.sqrt((patient_lat - hospital_lat)**2 + (patient_lon - hospital_lon)**2)  ## km
  eta = distance / 10 * 3600
  wait = (hospital.len_queue + np.sqrt(hospital.len_queue)) * hospital.mean_treat_time
  etaw = eta + wait
    
  return eta, etaw

def get_new_location(patient_location, hospital_location):
  patient_lon, patient_lat = patient_location
  hospital_lon, hospital_lat = hospital_location

  # Calculate the real distance using l2 norm
  distance = np.sqrt((patient_lat - hospital_lat)**2 + (patient_lon - hospital_lon)**2)  ## km
  #print("distance: {}".format(distance))
  distance_sec = 10 / 3600
  new_patient_lat = (patient_lat * (distance - distance_sec) + hospital_lat * distance_sec) / distance
  new_patient_lon = (patient_lon * (distance - distance_sec) + hospital_lon * distance_sec) / distance
  
  return (new_patient_lon, new_patient_lat)
  


  
def updater(model_type: str):  
  global patient_dict, hospitals_dict, tau, death_toll, treatment_toll, etaw_toll, etaw_dict
  """
  환자의 현재 위치 업데이트 (V)
  환자가 향하는 병원 업데이트 (V)
  사망 여부 확인 업데이트 (V)
  병원 도착 여부 확인 업데이트 (V)
  각 병원의 queue 업데이트 (V)
  각 병원의 치료 완료한 환자 수 업데이트 (V)  
  """

  """
  model_type: 'naive' or 'opt'
  """

  death_list = []
  arrival_list = []

  for id in patient_dict.keys():
    #################################################################
    ############# Step 1: 환자가 향해야 하는 병원 업데이트 #############
    # 처음에 환자가 생성되면, 해당 환자가 향해야 하는 병원을 optimizer를 통해 결정 (앞에서)
    # 여기서는 모든 환자가 1분마다 업데이트를 통해, 환자가 향해야 하는 병원을 다시 결정

    patient_dict[id].location = get_new_location(patient_dict[id].location, patient_dict[id].decision.location) # 이동하고 있는 환자의 위치를 새로이 정의

    if model_type == 'opt':
      if tau % 60 == 0: # 1분마다 업데이트 (의사결정 주기)
        curr_opt_hospital, new_etaw = optimizer(id) # 현재 환자가 향해야 하는 병원을 새로이 정의
        if patient_dict[id].decision != curr_opt_hospital:
          with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
              f.write(f"Time: {tau}\n")
              f.write(f"환자 {id}의 이동 경로가 변경되었습니다. from {patient_dict[id].decision.id} to {curr_opt_hospital.id}\n")
              f.write(f"환자 {id}의 예상 etaw: {new_etaw}\n")
          patient_dict[id].decision = curr_opt_hospital
    #################################################################
    ################# Step 2: 사망 여부 확인 업데이트 #################
    # 사망 여부 확인
    if tau - patient_dict[id].created_time > patient_dict[id].destiny:
      # etaw_toll += patient_dict[id].destiny[0] #$#$#$#$#$#$#$# 수정 #$#$#$#$#$#$#$#
      with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
          f.write(f"Time: {tau}\n")
          f.write(f"환자 {id}가 사망했습니다.\n")
      death_list.append(id)
      # 사망자 수 업데이트
      death_toll += 1
      continue

    #################################################################
    ############# Step 3: 병원 도착 여부 확인 업데이트 #############

    # 모든 환자가 각자 향하는 병원에 도착했는지 확인
    patient_lon, patient_lat = patient_dict[id].location
    #print("환자 {}의 현재 위치: {}".format(id, patient_dict[id].location))
    hospital_lon, hospital_lat = patient_dict[id].decision.location
    # 평면좌표계로 변환했으니, 0.1km = 100m 이내 도착시 도착으로 간주
    distance = np.sqrt((patient_lat - hospital_lat)**2 + (patient_lon - hospital_lon)**2)
    if distance < 0.01:
      # 환자의 treatment_time을 결정
      patient_dict[id].treatment_time = patient_dict[id].decision.sample_treat_time()      
      # 환자가 도착했다면, 환자를 patient_dict에서 삭제
      # 그 instance를 병원의 queue에 넣고, 병원의 len_queue를 1 증가
      patient_dict[id].decision.queue.append(patient_dict[id])
      patient_dict[id].decision.len_queue += 1
      if patient_dict[id].decision.treat_start == None:
        patient_dict[id].decision.treat_start = tau
      with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
          f.write(f"Time: {tau}\n")
          f.write(f"환자 {id}가 병원 {patient_dict[id].decision.id}에 도착했습니다.\n")
          f.write(f"환자 {id}가 병원 {patient_dict[id].decision.id}의 {patient_dict[id].decision.len_queue}번째 환자가 되었습니다.\n")
          f.write(f"환자 {id}의 치료 시간은 {patient_dict[id].treatment_time}입니다.\n")
      arrival_list.append(id)

  # 사망자 삭제
  for id in death_list:
    del patient_dict[id]

  # 도착자 삭제
  for id in arrival_list:
    del patient_dict[id]


  #################################################################
  ################# Step 4: 각 병원의 queue 업데이트 #################
  for key in hospitals_dict.keys():

    death_hospital_list = []
    treatment_hospital_list = []

    for i, patient in enumerate(hospitals_dict[key].queue[1:]):
      # 각 병원의 대기열 queue에 있는 환자들의 사망 여부 확인
      if tau - patient.created_time > patient.destiny:
        # etaw_toll += patient.destiny[0] #$#$#$#$#$#$#$# 수정 #$#$#$#$#$#$#$#        
        with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
            f.write(f"Time: {tau}\n")
            f.write(f"환자 {patient.id}가 사망했습니다.\n")
        death_hospital_list.append(patient)
        # 사망자 수 업데이트
        death_toll += 1
        # 병원의 len_queue를 1 감소
        hospitals_dict[key].len_queue -= 1
        continue
    # 각 병원의 대기열 queue에 있는 환자들의 치료 완료 여부 확인
    if hospitals_dict[key].len_queue > 0:
      if tau - hospitals_dict[key].treat_start > hospitals_dict[key].queue[0].treatment_time:
        with open(f'./log/log_{model_type}_{seed_num}.txt', 'a') as f:
            f.write(f"Time: {tau}\n")
            f.write(f"환자 {hospitals_dict[key].queue[0].id}가 치료를 완료했습니다.\n")
        # treatment_toll 업데이트
        treatment_toll += 1
        # etaw_toll 업데이트
        etaw_toll += tau - hospitals_dict[key].queue[0].created_time
        etaw_dict[hospitals_dict[key].queue[0].id] = tau - hospitals_dict[key].queue[0].created_time #### 수정
        treatment_hospital_list.append(hospitals_dict[key].queue[0])
        # 병원의 len_queue를 1 감소
        hospitals_dict[key].len_queue -= 1

        if hospitals_dict[key].len_queue > 0:
          hospitals_dict[key].treat_start = tau
        else:
          hospitals_dict[key].treat_start = None

    # 사망자 삭제
    for patient in death_hospital_list:
      hospitals_dict[key].queue.remove(patient)

    # 치료 완료자 삭제
    for patient in treatment_hospital_list:
      hospitals_dict[key].queue.remove(patient)


def naive_optimizer(patient_id):
  """
  Just return the hospital that is closest to the patient
  Input: patient_id
  Output: hospital_id (closest)
  """
  global patient_dict, hospitals_dict, tau
  
  patient = patient_dict[patient_id]

  min_distance = sys.maxsize
  min_hospital = None

  for hospital_id in hospitals_dict.keys():
    # Calculate the distance using l2 norm
    distance = np.sqrt((patient.location[0] - hospitals_dict[hospital_id].location[0])**2 + (patient.location[1] - hospitals_dict[hospital_id].location[1])**2)
    if distance < min_distance:
      min_distance = distance
      min_hospital = hospitals_dict[hospital_id]

  # Return min_hospital
  return min_hospital


def optimizer(patient_id):
  """
  Return the hospital that minimizes the ETAW for each patient
  Input: patient_id
  Output: hospital_id (minimize ETAW)
  """
  global patient_dict, hospitals_dict, tau
  
  patient = patient_dict[patient_id]

  min_etaw = sys.maxsize
  min_hospital = None

  for hospital_id in hospitals_dict.keys():
    eta, etaw = compute_etaw(patient, hospitals_dict[hospital_id], tau)
    if etaw < min_etaw:
      min_etaw = etaw
      min_hospital = hospitals_dict[hospital_id]

  # Return min_hospital
  return min_hospital, min_etaw

with open('./config.json') as f:
    config = json.load(f)
    
# model_type = input("Enter the model type: (opt or naive) ")

min_lat = config['min_lat']
max_lat = config['max_lat']
min_lon = config['min_lon']
max_lon = config['max_lon']

districts = config['districts'] # dict; {0: [(lon, lat), radius], ...}
# Transform the key of districts from str to int
districts = {int(key): districts[key] for key in districts.keys()}
weights = config['weights'] # list; population percentages
hospitals = config['hospitals'] # dict; {0: (lon, lat), ...}
# Transform the key of hospitals from str to int
hospitals = {int(key): hospitals[key] for key in hospitals.keys()}
mean_treat_times = config['mean_treat_times'] # list; average treatment time in each hospital
    
GRID = {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon}
patient_dict = {}  # dict; To store Patient instances
acc_patient_num = 0 # int; accumulated patient number

tau = 0 # Simulation global time step 
end_time = 24*3600*2 # Simulation end time

# Initialize the time points when patients occur on the map
patient_generate_mean = 60*5

patient_generate_points = []
patient_destiny_list = []
patient_district_list = []
patient_location_list = []
time_point = 0

alpha_skewed = 8 # Standard Alpha
beta_skewed = 3 # Standard Beta 
a_skewed = 0 # Min 생존 시간
b_skewed = 3600 # Max 생존 시간

while time_point < end_time:
    # np.random.seed(seed_num + time_point) ###################### 수정했음 (seed_num + time_point)
    sample = np.random.exponential(patient_generate_mean, 1).astype(int)[0]
    time_point += sample

    # np.random.seed(seed_num + time_point) ###################### 수정했음 (seed_num + time_point)
    d_id = np.random.choice(len(weights), p=weights)
    mean = districts[d_id][0]
    cov = [[districts[d_id][1]**2, 0], [0, districts[d_id][1]**2]]

    # np.random.seed(seed_num + time_point) ###################### 수정했음 (seed_num + time_point)
    lon, lat = np.random.multivariate_normal(mean, cov)
    lat = max(min(lat, GRID['max_lat']), GRID['min_lat'])
    lon = max(min(lon, GRID['max_lon']), GRID['min_lon'])
    location = (lon, lat)

    # np.random.seed(seed_num + time_point) ###################### 수정했음 (seed_num + time_point)
    destiny_temp = np.random.beta(alpha_skewed, beta_skewed, 1)
    destiny = a_skewed + (b_skewed - a_skewed) * destiny_temp
    
    if time_point < end_time:
        patient_generate_points.append(time_point)
        patient_destiny_list.append(destiny)
        patient_district_list.append(d_id)
        patient_location_list.append(location)
        

treatment_toll = 0
death_toll = 0
etaw_toll = 0
patient_toll = len(patient_generate_points)

hospitals_dict = {}
for id in range(len(hospitals)):
    hospitals_dict[id] = Hospital(id, hospitals[id], mean_treat_times[id], seed_num)
    
metrics_df = pd.DataFrame(columns=['treatment_toll', 'death_toll', 'etaw_toll', 'patient_toll'])

# ETAW_dict to store the etaw of each patient
etaw_dict = dict()

def draw_cur_map():
    global tau, patient_dict, hospitals, min_lat, max_lat, min_lon, max_lon

    # Plot the current map
    plt.figure(figsize=(10, 10))
    m = Basemap(projection='merc', llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='white', lake_color='white')
    m.drawparallels(np.arange(min_lat, max_lat, 0.1), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(min_lon, max_lon, 0.1), labels=[0, 0, 0, 1])
    # Plot the hospitals
    for id in range(len(hospitals)):
        x, y = m(hospitals[id][0], hospitals[id][1])
        m.plot(x, y, 'bo', markersize=5)
    # Plot the patients
    for id in patient_dict.keys():
        x, y = m(patient_dict[id].location[0], patient_dict[id].location[1])
        m.plot(x, y, 'ro', markersize=5)
    plt.title("Time: {}".format(tau))
    plt.savefig('./images/{}.png'.format(tau))    


for tau in tqdm(range(0, end_time)):  # start_time should be defined, or use 0 if it starts from the beginning
    if tau in patient_generate_points:
        # print("Time: {}".format(tau))
        create_patient(model_type, patient_location_list[0], patient_district_list[0], patient_destiny_list[0])  # Create patient
        patient_location_list.pop(0)
        patient_district_list.pop(0)
        patient_destiny_list.pop(0)
    updater(model_type)  # Update patient's location + alpha
    
    # Add the current metrics to the metrics_df
    metrics_df.loc[tau] = [treatment_toll, death_toll, etaw_toll, patient_toll]


# Save the metrics_df
if model_type == 'opt':
    # Check if the directory exists
    if not os.path.exists('./opt_metrics'):
        os.makedirs('./opt_metrics')
    metrics_df.to_csv(f'./opt_metrics/metrics_df_opt_{seed_num}.csv')
elif model_type == 'naive':
    # Check if the directory exists
    if not os.path.exists('./naive_metrics'):
        os.makedirs('./naive_metrics')
    metrics_df.to_csv(f'./naive_metrics/metrics_df_naive_{seed_num}.csv')

# Save the etaw_dict
if model_type == 'opt':
    # Check if the directory exists
    if not os.path.exists('./opt_etaw'):
        os.makedirs('./opt_etaw')
    with open(f'./opt_etaw/etaw_dict_opt_{seed_num}.json', 'w') as f:
        json.dump(etaw_dict, f)
elif model_type == 'naive':
    # Check if the directory exists
    if not os.path.exists('./naive_etaw'):
        os.makedirs('./naive_etaw')
    with open(f'./naive_etaw/etaw_dict_naive_{seed_num}.json', 'w') as f:
        json.dump(etaw_dict, f)