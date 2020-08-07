import gzip
import json
import os
import hashlib
from pprint import pprint as pp
from itertools import combinations
import datetime
import numpy as np

# Python 2
#import urllib2

# Python 3
import urllib
import util_5353

BASE_URL = 'https://syntheticmass.mitre.org/fhir/'
MAX_PATIENTS = 50000
CACHE_FILE = 'cache.dat'
PATH_CACHE = {}

# Returns the JSON result at the given URL.  Caches the results so we don't
# unnecessarily hit the FHIR server.  Note this ain't the best caching, as
# it's going to save a bunch of tiny files that could probably be handled more
# efficiently.
def get_url(url):
  # First check the cache
  if len(PATH_CACHE) == 0:
    for line in open(CACHE_FILE).readlines():
      split = line.strip().split('\t')
      cached_path = split[0]
      cached_url = split[1]
      PATH_CACHE[cached_url] = cached_path
  if url in PATH_CACHE:
    path = PATH_CACHE[url].replace('?', '_')
    return json.loads(gzip.open(path).read().decode('utf-8'))

  print('Retrieving:', url)

  print('You are about to query the FHIR server, which probably means ' + \
        'that you are doing something wrong.  But feel free to comment ' + \
        'out this bit of code and proceed right ahead.')
  exit(1)
  print('Note: the code below is not tested for Python 3, you will likely ' + \
        'need to make a few changes, e.g., urllib2')

  resultstr = urllib.urlopen(url).read()
  json_result = json.loads(resultstr)

  # Remove patient photos, too much space
  if url.replace(BASE_URL, '').startswith('Patient'):
    for item in json_result['entry']:
      item['resource']['photo'] = 'REMOVED'

  m = hashlib.md5()
  m.update(url)
  md5sum = m.hexdigest()

  path_dir = 'cache/' + md5sum[0:2] + '/' + md5sum[2:4] + '/'
  if not os.path.exists('cache'):
    os.mkdir('cache')
  if not os.path.exists('cache/' + md5sum[0:2]):
    os.mkdir('cache/' + md5sum[0:2])
  if not os.path.exists(path_dir):
    os.mkdir(path_dir)
  path = path_dir + url.replace(BASE_URL, '')

  w = gzip.open(path, 'wb')
  w.write(json.dumps(json_result))
  w.close()
  w = open(CACHE_FILE, 'a')
  w.write(path + '\t' + url + '\n')
  w.close()
  PATH_CACHE[url] = path

  return json_result

# For pagination, returns the next URL
def get_next(result):
  links = result['link']
  for link in links:
    if link['relation'] == 'next':
      return link['url']

# Returns the list of patients based on the given filter
#def get_patients(pt_filter):
#  patients = []
#  url = BASE_URL + 'Patient?_offset=0&_count=1000'
#  while url is not None:
#    patients_page = get_url(url)
#    if 'entry' not in patients_page:
#      break
#    for patient_json in patients_page['entry']:
#      if pt_filter.include(patient_json['resource']):
#        patients.append(patient_json['resource'])
#        if MAX_PATIENTS is not None and len(patients) == MAX_PATIENTS:
#          return patients
#    url = get_next(patients_page)
#  return patients
def get_patients(pt_filter):
  patients = []
  url = BASE_URL + 'Patient?_offset=0&_count=1000'
  while url is not None:
    patients_page = get_url(url)
    if 'entry' not in patients_page:
      break
    for patient_json in patients_page['entry']:
      patients.append(patient_json['resource'])
      if MAX_PATIENTS is not None and len(patients) == MAX_PATIENTS:
        return [p for p in patients if pt_filter.include(p)]
    url = get_next(patients_page)
  return [p for p in patients if pt_filter.include(p)]

# Returns the conditions for the patient with the given patient_id
def get_conditions(patient_id):
  url = BASE_URL + 'Condition?patient=' + patient_id + '&_offset=0&_count=1000'
  conditions = []
  while url is not None:
    conditions_page = get_url(url)
    if 'entry' in conditions_page:
      conditions.extend([c['resource'] for c in conditions_page['entry']])
    url = get_next(conditions_page)
  return conditions

# Returns the observations for the patient with the given patient_id
def get_observations(patient_id):
  url = BASE_URL + 'Observation?patient=' + patient_id + '&_offset=0&_count=1000'
  observations = []
  while url is not None:
    observations_page = get_url(url)
    if 'entry' in observations_page:
      observations.extend([o['resource'] for o in observations_page['entry']])
    url = get_next(observations_page)
  return observations

# Returns the medications for the patient with the given patient_id
def get_medications(patient_id):
  url = BASE_URL + 'MedicationRequest?patient=' + patient_id + '&_offset=0&_count=1000'
  medications = []
  DBG = 0
  while url is not None:
    medications_page = get_url(url)
    if 'entry' in medications_page:
      medications.extend([c['resource'] for c in medications_page['entry']])
    url = get_next(medications_page)
  return medications

# Problem 1 [10 points]
def num_patients(pt_filter):
  tup = None
  # Begin CODE
  patients = get_patients(pt_filter)
  total_patients = len(patients)
  surnames = []
  for patient in patients:
    for name in patient['name']:
      if name['use'] == 'official' and name['family'] not in surnames:
        surnames.append(name['family'])
  tup = (total_patients, len(surnames))
  # End CODE
  return tup

# Takes a dictionary of a codeable concept (http://hl7.org/fhir/datatypes.html#CodeableConcept) 
# Return the code if it exists, otherwise returns None
def get_code(cc, systems):
  if not cc:
    return None

  codings = cc.get('coding')
  if(codings):
    for coding in codings:
        if coding.get('system') in systems:
          return coding.get('code')
  return None

# Takes a dictionary of a codeable concept (http://hl7.org/fhir/datatypes.html#CodeableConcept) 
# Return the display name if it exists, otherwise returns None
def get_display(cc, systems):
  if not cc:
    return None

  codings = cc.get('coding')
  if(codings):
    for coding in codings:
      if coding.get('system') in systems:
        return coding.get('display')
  return None

# Problem 2 [10 points]
def patient_stats(pt_filter):
  stats = {}
  # Begin CODE
  patients = get_patients(pt_filter)
  attributes = ['gender', 'marital_status', 'race', 'ethnicity', 'age', 'with_address']

  for attribute in attributes:
    frequencies = {}

    for patient in patients:

      if attribute == 'gender':
        gender = patient.get('gender', 'UNK')
        if gender not in frequencies:
          frequencies[gender] = 1
        else:
          frequencies[gender] += 1

      elif attribute == 'marital_status':
        marital_status = get_code(patient.get('maritalStatus'), ['http://loinc.org'])
        if not marital_status:
          marital_status = 'UNK'

        if marital_status not in frequencies:
          frequencies[marital_status] = 1
        else:
          frequencies[marital_status] += 1

      elif attribute in ['race', 'ethnicity']:
        found = False;
        extended_attributes = patient.get('extension')
        if extended_attributes:
          for extended_attribute in extended_attributes:
            if 'valueCodeableConcept' in extended_attribute and extended_attribute['valueCodeableConcept'].get('text') == attribute:
              attr = None
              cc = extended_attribute.get('valueCodeableConcept')
              attr = get_display(cc, ['http://hl7.org/fhir/v3/Race', 'http://hl7.org/fhir/v3/Ethnicity'])

              if attr:
                found = True
              else:
                attr = 'UNK'

              if attr not in frequencies:
                frequencies[attr] = 1
              else:
                frequencies[attr] += 1
              
              break

        if found == False:
          if 'UNK' not in frequencies:
            frequencies['UNK'] = 1
          else:
            frequencies['UNK'] += 1

      elif attribute == 'age':
        birthDate = patient.get('birthDate', 'UNK')
        year = birthDate
        if birthDate:
          year = int(birthDate[0:4])
          year = year - year % 10
          year = str(year)
        if year not in frequencies:
          frequencies[year] = 1
        else:
          frequencies[year] += 1

      elif attribute == 'with_address':
        has_address = 'address' in patient
        has_address_string = ''
        if has_address:
          has_address_string = 'yes_address'
        else:
          has_address_string = 'no_address'

        if has_address_string not in frequencies:
          frequencies[has_address_string] = 1
        else:
          frequencies[has_address_string] += 1

    probabilities = {}
    for k,v in frequencies.items():
      probabilities.update( { k: v/sum( frequencies.values() ) } )
    stats[attribute] = probabilities
    '''pp(frequencies)
    print(sum(frequencies.values()))
    pp(probabilities)'''

  # End CODE
  return stats

# Problem 3 [15 points]
def diabetes_quality_measure(pt_filter):
  tup = None
  # Begin CODE
  total_diabetes = 0
  has_test = 0      # total number of diabetes patients that have at least one hemoglobin a1c test
  has_test_gt6 = 0  # total number of diabetes patients that have at least one hemoglobin a1c test greater than 6.0

  patients = get_patients(pt_filter)
  #pp(patients[0])
  for patient in patients:
    id = patient['id']
    conditions = get_conditions(id)
    for condition in conditions:
      code = get_code(condition.get('code'), ['http://snomed.info/sct'])
      if code == '44054006':
        total_diabetes += 1

        observations = get_observations(id)
        for observation in observations:
          code = get_code(observation.get('code'), ['http://loinc.org'])
          if code == '4548-4':
            has_test += 1
            break

        for observation in observations:
          code = get_code(observation.get('code'), ['http://loinc.org'])
          if code == '4548-4':
            value = observation.get('valueQuantity').get('value')
            value = float(value)
            if value > 6:
              has_test_gt6 += 1
              break


  tup = ( total_diabetes, has_test, has_test_gt6 )
  #print(tup)
  # End CODE
  return tup

# Problem 4 [10 points]
def common_condition_pairs(pt_filter):
  pairs = []
  # Begin CODE
  co_ocurring_conditions = {}
  patients = get_patients(pt_filter)

  for patient in patients:
    id = patient['id']
    conditions = get_conditions(id)
    active_conditions = []

    for condition in conditions:
      name = get_display( condition.get('code'), ['http://snomed.info/sct'] )
      if name and condition.get('clinicalStatus') == 'active' and name not in active_conditions:
        active_conditions.append(name)

    active_conditions.sort()
    if len(active_conditions) >= 2:
      for combination in combinations(active_conditions,2):
        if combination not in co_ocurring_conditions:
          co_ocurring_conditions[combination] = 1
        else: co_ocurring_conditions[combination] += 1

  top10 = sorted(co_ocurring_conditions.items(), key=lambda x: x[1], reverse=True)[0:10]
  #print(top10)
  pairs = [ i[0] for i in top10 ]
  #print(pairs)

  # End CODE
  return pairs

# Problem 5 [10 points]
def common_medication_pairs(pt_filter):
  pairs = []
  # Begin CODE
  medication_pairs = {}
  patients = get_patients(pt_filter)

  for patient in patients:
    id = patient['id']
    medications = get_medications(id)

    active_medications = []

    for medication in medications:
      name = get_display( medication.get('medicationCodeableConcept'), ['http://www.nlm.nih.gov/research/umls/rxnorm'] )
      if name and medication.get('status') == 'active' and name not in active_medications:
        active_medications.append(name)

    active_medications.sort()
    if len(active_medications) >= 2:
      for combination in combinations(active_medications, 2):
        if combination not in medication_pairs:
          medication_pairs[combination] = 1
        else:
          medication_pairs[combination] += 1
    
  top10 = sorted(medication_pairs.items(), key=lambda x: x[1], reverse=True)[0:10]
  #pp(top10)
  pairs = [ i[0] for i in top10 ]
  #pp(pairs)

  # End CODE
  return pairs

# Problem 6 [10 points]
def conditions_by_age(pt_filter):
  tup = None
  # Begin CODE
  currentDate = datetime.date.fromisoformat('2018-01-31')
  fifty_years = datetime.timedelta(days=50*365)
  fifteen_years = datetime.timedelta(days=15*365)

  patients = get_patients(pt_filter)

  older_conditions = {} # active, non-inflammation conditions, patients older than 50
  younger_conditions = {} # active, non-inflammation conditions, patients younger than 15

  for patient in patients:
    birthDateStr = patient.get('birthDate')
    if not birthDateStr:
      continue
    birthDate = datetime.date.fromisoformat(birthDateStr)
    if not birthDate:
      continue
    
    id = patient['id']
    conditions = get_conditions(id)
    active_conditions = []

    for condition in conditions:
      name = get_display( condition.get('code'), ['http://snomed.info/sct'] )
      if name and condition.get('clinicalStatus') == 'active' and name not in active_conditions and not 'itis' in name:
        active_conditions.append(name)

    for condition in active_conditions:
      if birthDate <= currentDate - fifty_years:
        if condition not in older_conditions:
          older_conditions[condition] = 1
        else:
          older_conditions[condition] += 1
      elif currentDate - fifteen_years <= birthDate:
        if condition not in younger_conditions:
          younger_conditions[condition] = 1
        else:
          younger_conditions[condition] += 1

  older = sorted(older_conditions.items(), key=lambda x: x[0], reverse=False)
  younger = sorted(younger_conditions.items(), key=lambda x: x[0], reverse=False)
  older.sort(key=lambda x: x[1], reverse=True)
  younger.sort(key=lambda x: x[1], reverse=True)

  #pp(older[0:10])
  #pp(younger[0:10])
  
  tenold = [ i[0] for i in older[0:10] ]
  tenyoung = [ i[0] for i in younger[0:10] ]

  #pp(tenold)
  #pp(tenyoung)

  tup = ( tenold, tenyoung )
  #pp(tup)
  # End CODE
  return tup

# Problem 7 [10 points]
def medications_by_gender(pt_filter):
  tup = None
  # Begin CODE
  medications_male = {}
  medications_female = {}

  patients = get_patients(pt_filter)

  for patient in patients:
    #pp(patient)
    active_medications = []
    id = patient['id']
    medications = get_medications(id)
    for medication in medications:
      #pp(medication)
      name = get_display( medication.get('medicationCodeableConcept'), ['http://www.nlm.nih.gov/research/umls/rxnorm'])
      if name and medication.get('status') == 'active' and name not in active_medications:
        active_medications.append(name)

    for medication in active_medications:
      if patient.get('gender') == 'female':
        if medication not in medications_female:
          medications_female[medication] = 1
        else:
          medications_female[medication] += 1
      elif patient.get('gender') == 'male':
        if medication not in medications_male:
          medications_male[medication] = 1
        else:
          medications_male[medication] += 1

  male_meds = sorted(medications_male.items(), key=lambda x: x[0], reverse=False)
  female_meds = sorted(medications_female.items(), key=lambda x: x[0], reverse=False)
  male_meds.sort(key=lambda x: x[1], reverse=True)
  female_meds.sort(key=lambda x: x[1], reverse=True)
  
  #pp(male_meds[0:10])
  #pp(female_meds[0:10])

  male = [ i[0] for i in male_meds[0:10] ]
  female = [ i[0] for i in female_meds[0:10] ]
  tup = (male, female)
  #pp(tup)
  # End CODE
  return tup

# Problem 8 [25 points]
def bp_stats(pt_filter):
  stats = []
  # Begin CODE
  normal_conditions = []
  abnormal_conditions = []
  no_readings_conditions = []

  normal_readings = {}
  abnormal_readings = {}
  no_readings = {}

  patients = get_patients(pt_filter)

  for patient in patients:
    id = patient['id']
    observations = get_observations(id)
    conditions = get_conditions(id)

    systolic_readings = []
    diastolic_readings = []

    for observation in observations:
      #pp(observation)
      code = get_code( observation.get('code'), ['http://loinc.org'])
      if code == '55284-4':
        #pp(observation)
        for component in observation.get('component'):
          #pp(component)
          measurement_name = get_display( component.get('code'), ['http://loinc.org'] )
          #print(measurement_name)
          if measurement_name == 'Systolic Blood Pressure':
            bp = component.get('valueQuantity')
            if bp:
              bp = bp.get('value')
            if bp:
              bp = float(bp)
              systolic_readings.append(bp)
          elif measurement_name == 'Diastolic Blood Pressure':
            bp = component.get('valueQuantity')
            if bp:
              bp = bp.get('value')
            if bp:
              bp = float(bp)
              diastolic_readings.append(bp)

    normal = 0
    abnormal = 0
    #print(len(systolic_readings), len(diastolic_readings))
    for s,d in zip(systolic_readings, diastolic_readings):
      if (90 <= s and s <= 140) and (60 <= d and d <= 90):
        normal += 1
      else:
        abnormal += 1
    #print(normal, abnormal)

    num_conditions = len(conditions)
    #print(num_conditions)
    total = normal + abnormal
    if( total == 0 ):
      no_readings_conditions.append(num_conditions)
    elif( normal/total >= 0.9 ):
      normal_conditions.append(num_conditions)
    elif( abnormal/total > 0.1 ):
      abnormal_conditions.append(num_conditions)

  stats = 'min, max, median, mean, stddev'.split(', ')
  functions = [min, max, np.median, np.mean, np.std]
  for stat,func in zip(stats, functions):
    abnormal_readings[stat] = func(abnormal_conditions)
    normal_readings[stat] = func(normal_conditions)
    no_readings[stat] = func(no_readings_conditions)

  stats = [normal_readings, abnormal_readings, no_readings]

  #pp(stats)

  # End CODE
  return stats


# Basic filter, lets everything pass
class all_pass_filter:
  def id(self):
    return 'all_pass'
  def include(self, patient):
    util_5353.assert_dict_key(patient, 'id', 'pt_filter')
    util_5353.assert_dict_key(patient, 'name', 'pt_filter')
    util_5353.assert_dict_key(patient, 'address', 'pt_filter')
    util_5353.assert_dict_key(patient, 'birthDate', 'pt_filter')
    util_5353.assert_dict_key(patient, 'gender', 'pt_filter')
    return True

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  # Include all patients
  pt_filter = all_pass_filter()

  print('::: Problem 1 :::')
  one_ret = num_patients(pt_filter)
  util_5353.assert_tuple(one_ret, 2, '1')
  util_5353.assert_int_range((0, 10000000), one_ret[0], '1')
  util_5353.assert_int_range((0, 10000000), one_ret[1], '1')

  print('::: Problem 2 :::')
  two_ret = patient_stats(pt_filter)
  util_5353.assert_dict(two_ret, '2')
  for key in ['gender', 'marital_status', 'race', 'ethnicity', 'age', 'with_address']:
    util_5353.assert_dict_key(two_ret, key, '2')
    util_5353.assert_dict(two_ret[key], '2')
    for key2 in two_ret[key].keys():
      util_5353.assert_str(key2, '2')
    util_5353.assert_prob_dict(two_ret[key], '2')
  for key2 in two_ret['age'].keys():
    if not key2.isdigit():
      util_5353.die('2', 'age key should be year: %s', key2)

  print('::: Problem 3 :::')
  three_ret = diabetes_quality_measure(pt_filter)
  util_5353.assert_tuple(three_ret, 3, '3')
  util_5353.assert_int_range((0, 1000000), three_ret[0], '3')
  util_5353.assert_int_range((0, 1000000), three_ret[1], '3')
  util_5353.assert_int_range((0, 1000000), three_ret[2], '3')
  if three_ret[0] < three_ret[1] or three_ret[1] < three_ret[2]:
    util_5353.die('3', 'Values should be in %d >= %d >= %d', three_ret)

  print('::: Problem 4 :::')
  four_ret = common_condition_pairs(pt_filter)
  util_5353.assert_list(four_ret, 10, '4')
  for i in range(len(four_ret)):
    util_5353.assert_tuple(four_ret[i], 2, '4')
    util_5353.assert_str(four_ret[i][0], '4')
    util_5353.assert_str(four_ret[i][1], '4')

  print('::: Problem 5 :::')
  five_ret = common_medication_pairs(pt_filter)
  util_5353.assert_list(five_ret, 10, '5')
  for i in range(len(five_ret)):
    util_5353.assert_tuple(five_ret[i], 2, '5')
    util_5353.assert_str(five_ret[i][0], '5')
    util_5353.assert_str(five_ret[i][1], '5')

  print('::: Problem 6 :::')
  six_ret = conditions_by_age(pt_filter)
  util_5353.assert_tuple(six_ret, 2, '6')
  util_5353.assert_list(six_ret[0], 10, '6')
  util_5353.assert_list(six_ret[1], 10, '6')
  for i in range(len(six_ret[0])):
    util_5353.assert_str(six_ret[0][i], '6')
    util_5353.assert_str(six_ret[1][i], '6')

  print('::: Problem 7 :::')
  seven_ret = medications_by_gender(pt_filter)
  util_5353.assert_tuple(seven_ret, 2, '6')
  util_5353.assert_list(seven_ret[0], 10, '6')
  util_5353.assert_list(seven_ret[1], 10, '6')
  for i in range(len(seven_ret[0])):
    util_5353.assert_str(seven_ret[0][i], '6')
    util_5353.assert_str(seven_ret[1][i], '6')

  print('::: Problem 8 :::')
  eight_ret = bp_stats(pt_filter)
  util_5353.assert_list(eight_ret, 3, '8')
  for i in range(len(eight_ret)):
    util_5353.assert_dict(eight_ret[i], '8')
    util_5353.assert_dict_key(eight_ret[i], 'min', '8')
    util_5353.assert_dict_key(eight_ret[i], 'max', '8')
    util_5353.assert_dict_key(eight_ret[i], 'median', '8')
    util_5353.assert_dict_key(eight_ret[i], 'mean', '8')
    util_5353.assert_dict_key(eight_ret[i], 'stddev', '8')

  print('~~~ All Tests Pass ~~~')


