
import util_5353
import lxml.etree
import editdistance
from sklearn.linear_model import ElasticNetCV
from pprint import pprint as pp

MEDICATIONS = ['remeron', 'lexapro', 'effexor', 'zoloft', 'celexa', 
               'wellbutrin', 'paxil', 'savella', 'prozac', 'cymbalta']
MUTATIONS = ['chrom_1.pos_98539.ref_A.alt_T',  'chrom_1.pos_88327.ref_C.alt_A',
             'chrom_1.pos_63872.ref_C.alt_T',  'chrom_1.pos_96696.ref_A.alt_G',
             'chrom_2.pos_97561.ref_G.alt_A',  'chrom_2.pos_69421.ref_A.alt_C',
             'chrom_2.pos_70704.ref_G.alt_A',  'chrom_2.pos_30517.ref_A.alt_C',
             'chrom_3.pos_57245.ref_G.alt_A',  'chrom_3.pos_64337.ref_T.alt_C',
             'chrom_3.pos_48160.ref_A.alt_C',  'chrom_3.pos_14811.ref_G.alt_A',
             'chrom_4.pos_99335.ref_T.alt_C',  'chrom_4.pos_49304.ref_G.alt_T',
             'chrom_4.pos_93162.ref_A.alt_T',  'chrom_4.pos_35883.ref_A.alt_G',
             'chrom_5.pos_99641.ref_T.alt_A',  'chrom_5.pos_47810.ref_T.alt_A',
             'chrom_5.pos_41351.ref_T.alt_C',  'chrom_5.pos_30106.ref_A.alt_C',
             'chrom_6.pos_95091.ref_C.alt_G',  'chrom_6.pos_22806.ref_C.alt_G',
             'chrom_6.pos_6035.ref_T.alt_A',   'chrom_6.pos_57950.ref_A.alt_G',
             'chrom_7.pos_66842.ref_C.alt_A',  'chrom_7.pos_40665.ref_C.alt_T',
             'chrom_7.pos_16241.ref_T.alt_A',  'chrom_7.pos_46163.ref_T.alt_A',
             'chrom_8.pos_93350.ref_A.alt_G',  'chrom_8.pos_73332.ref_T.alt_G',
             'chrom_8.pos_17571.ref_C.alt_A',  'chrom_8.pos_92636.ref_C.alt_G',
             'chrom_9.pos_99676.ref_G.alt_A',  'chrom_9.pos_14535.ref_C.alt_A',
             'chrom_9.pos_35056.ref_A.alt_G',  'chrom_9.pos_28381.ref_A.alt_G',
             'chrom_10.pos_54525.ref_C.alt_G', 'chrom_10.pos_87597.ref_T.alt_A',
             'chrom_10.pos_54127.ref_G.alt_T', 'chrom_10.pos_13058.ref_C.alt_A']

# Problem A [0 points]
def read_data(filename, attempt):
  data = None
  # Begin CODE
  file = open('depression_study.1000.xml')
  data = lxml.etree.parse(file).getroot()
  # End CODE
  return data

# Problem 1 [10 points]
def mean_missed_reports(data, attempt):
  mean = None
  # Begin CODE
  counter=0
  for result in data.iter('Result'):
    if result.get('medication') == 'none':
      counter+=1
  pt_counter=0
  for pt in data.iter('Patient'):
      pt_counter+=1
  mean=counter/pt_counter
  # End CODE
  return mean

# Problem 2 [10 points]
def total_medication_misspellings(data, attempt):
  total = 0
  # Begin CODE
  for result in data.iter('Result'):
    if result.get('medication') not in MEDICATIONS and not result.get('medication')=='none':
      total+=1
  # End CODE
  return total

def clean_name(name, valid_names):
  distances = []
  for vn in valid_names:
    distances.append( editdistance.eval(name, vn) )
  mindist = min(distances)
  index = distances.index(mindist)
  truename = valid_names[index]
  return truename

# Problem 3 [10 points]
def medications_by_frequency(data, attempt):
  medications = []
  # Begin CODE
  freqs = dict( [(name, 0) for name in MEDICATIONS] )
  for patient in data:
    results = patient.findall('Results/Result')
    result_tuples = []
    for result in results:
      date = result.get('date')
      medname = result.get('medication')
      if medname != 'none':
        medname = clean_name( medname, MEDICATIONS)
      result_tuples.append( (int(date), medname ) )
    schedule = sorted( result_tuples, key= lambda x: x[0] )

    cur = schedule[0][1]
    if cur in MEDICATIONS:
      freqs[cur] += 1
    for i in range( 1, len(schedule) ):
      last = cur
      cur = schedule[i][1]
      if last != cur and cur in MEDICATIONS:
        freqs[cur] += 1

  res = sorted(freqs.items(), key = lambda x:x[1], reverse=True)
  medications=([pair[0] for pair in res])
  # End CODE
  return medications

# Problem 4 [10 points]
def total_mutation_corruptions(data, attempt):
  total = 0
  # Begin CODE
  for mutation in data.iter('Mutation'):
    chromosome=mutation.get('chromosome')
    pos=mutation.get('pos')
    ref=mutation.get('ref')
    alt=mutation.get('alt')
    s=f'chrom_{chromosome}.pos_{pos}.ref_{ref}.alt_{alt}'
    if s not in MUTATIONS:
      total+=1
  # End CODE
  return total

# Problem 5 [10 points]
def mutations_by_frequency(data, attempt):
  mutations = []
  # Begin CODE
  freqs = dict( [(name, 0) for name in MUTATIONS] )
  for mutation in data.iter('Mutation'):
    chromosome=mutation.get('chromosome')
    pos=mutation.get('pos')
    ref=mutation.get('ref')
    alt=mutation.get('alt')
    s=f'chrom_{chromosome}.pos_{pos}.ref_{ref}.alt_{alt}'

    if s not in MUTATIONS:
      distances = []
      for name in MUTATIONS:
        distances.append( editdistance.eval(s, name) )
      mindist = min(distances)
      index = distances.index(mindist)
      truename = MUTATIONS[index]
      freqs[truename]+=1
  res= sorted(freqs.items(), key = lambda x:x[1], reverse=True)
  mutations=([pair[0] for pair in res])
  # End CODE
  return mutations

# Problem 6 [20 points]
def mutation_impact(data, attempt):
  impact = {m:None for m in MUTATIONS}
  # Begin CODE
  Y = []
  X = []

  for patient in data:
    hamd = patient.get('baseline_hamd')
    Y.append(hamd)
    patient_mutations = []
    for mutation in patient.iter('Mutation'):
      chromosome=mutation.get('chromosome')
      pos=mutation.get('pos')
      ref=mutation.get('ref')
      alt=mutation.get('alt')
      s=f'chrom_{chromosome}.pos_{pos}.ref_{ref}.alt_{alt}'
      name = clean_name(s, MUTATIONS)
      patient_mutations.append(name)
    patient_X = []
    for mutation in MUTATIONS:
      if mutation in patient_mutations:
        patient_X.append(1)
      else:
        patient_X.append(0)
    X.append(patient_X)

  reg = ElasticNetCV()
  reg.fit(X,Y)
  for i,m in enumerate(MUTATIONS):
    impact[m] = float(reg.coef_[i])
    
  # End CODE
  return impact

# Problem 7 [10 points]
def mutation_medication_impact(data, attempt):
  impact = {m:{med:None for med in MEDICATIONS} for m in MUTATIONS}
  # Begin CODE
  X = []
  Y = []
  for patient in data:
    baseline_hamd = int( patient.get('baseline_hamd') )

    patient_mutations = []
    for mutation in patient.iter('Mutation'):
      chromosome=mutation.get('chromosome')
      pos=mutation.get('pos')
      ref=mutation.get('ref')
      alt=mutation.get('alt')
      s=f'chrom_{chromosome}.pos_{pos}.ref_{ref}.alt_{alt}'
      name = clean_name(s, MUTATIONS)
      patient_mutations.append(name)

    full_medication_schedule = []
    results = patient.findall('Results/Result')
    schedule = []
    medication_dict = {}
    for result in results:
      date = result.get('date')
      medname = result.get('medication')
      if medname != 'none':
        medname = clean_name( medname, MEDICATIONS)
      curhamd = result.get('hamd')
      schedule.append( (int(date), medname, int(curhamd) ) )
    schedule = sorted( schedule, key= lambda x: x[0])

    cur = schedule[0][1]
    curdate = schedule[0][0]
    curhamd = schedule[0][2]
    if cur != 'none':
      medication_dict[cur] = ( curdate, 0.5, curhamd - baseline_hamd )
      full_medication_schedule.append( medication_dict.copy() )
    for i in range( 1, len(schedule) ):

      last = cur
      lastdate = curdate
      cur = schedule[i][1]
      curdate = schedule[i][0]
      curhamd = schedule[i][2]

      if last == cur and cur != 'none':       # If same and neither none
        medication_dict[cur] = ( curdate, 1, curhamd - baseline_hamd )
      elif last != cur and last != 'none' and cur != 'none':
        medication_dict = {}                  # If change and neither none
        medication_dict[cur] = (curdate, 0.5, curhamd - baseline_hamd )
        if curdate - lastdate == 1:
          medication_dict[last] = ( curdate, 0.5, curhamd - baseline_hamd )
      elif last == 'none' and cur == 'none':  # If both none
        medication_dict = {}
      elif last != 'none' and cur == 'none':  # If change to none
        medication_dict[last] = (curdate, 0.5, curhamd - baseline_hamd )
      elif last == 'none' and cur != 'none':  # If change from none
        medication_dict = {}
        medication_dict[cur] = ( curdate, 0.5, curhamd - baseline_hamd )

      full_medication_schedule.append( medication_dict.copy() )
    
    patientXs = []
    patientYs = []
    for medication_schedule in full_medication_schedule:
      if medication_schedule != {}:
        hamd = max( medication_schedule.items(), key= lambda x: x[1][0] )[1][2]
        patientYs.append(hamd)
        curX = []
        for mutation in MUTATIONS:
          for medication in MEDICATIONS:
            if mutation in patient_mutations and medication in medication_schedule:
              curX.append( medication_schedule[medication][1] )
            else:
              curX.append(0)
        patientXs.append(curX)

    X += patientXs
    Y += patientYs

  reg = ElasticNetCV()
  reg.fit(X,Y)
  print(reg.coef_)
  allcombos = []
  for mutation in MUTATIONS:
    for medication in MEDICATIONS:
      allcombos.append( (mutation, medication) )
  for i, (mutation, medication) in enumerate(allcombos):
    impact[mutation][medication] = float( reg.coef_[i] )
  
  print(impact)

  # End CODE
  return impact

# Problem 8 [20 points]
def medication_impact(data, attempt):
  medications = {med:None for med in MEDICATIONS}
  # Begin CODE

  # End CODE
  return medications

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  participants = 10000
  attempts = [1, 2, 3]   # change to [1] or [1, 2] if appropriate
  
  for attempt in attempts:
    print('::  ATTEMPT %d ::' % attempt)
    print('::: Problem 0 :::')
    data = read_data('depression_study.' + str(participants) + '.xml', attempt)

    print('::: Problem 1 :::')
    one_ret = mean_missed_reports(data, attempt)
    util_5353.assert_float(one_ret, '1')
    util_5353.assert_float_range((0.01, 26.0), one_ret, '1')

    print('::: Problem 2 :::')
    two_ret = total_medication_misspellings(data, attempt)
    util_5353.assert_int(two_ret, '2')
    util_5353.assert_int_range((1, 26 * participants), two_ret, '2')

    print('::: Problem 3 :::')
    three_ret = medications_by_frequency(data, attempt)
    util_5353.assert_list(three_ret, len(MEDICATIONS), '3', valid_values=MEDICATIONS)

    print('::: Problem 4 :::')
    four_ret = total_mutation_corruptions(data, attempt)
    util_5353.assert_int(two_ret, '4')
    util_5353.assert_int_range((1, 40 * participants), two_ret, '4')

    print('::: Problem 5 :::')
    five_ret = mutations_by_frequency(data, attempt)
    util_5353.assert_list(five_ret, None, '5', valid_values=MUTATIONS)
    util_5353.assert_int_range((1, 40), len(five_ret), '5')
    util_5353.assert_int_eq(len(five_ret), len(set(five_ret)), '5')

    print('::: Problem 6 :::')
    six_ret = mutation_impact(data, attempt)
    util_5353.assert_dict(six_ret, '6')
    util_5353.assert_int_eq(len(MUTATIONS), len(six_ret), '6')
    for k,v in six_ret.items():
      util_5353.assert_str(k, '6', valid_values=MUTATIONS)
      util_5353.assert_float(v, '6')
      util_5353.assert_float_range((-10.0, 10.0), v, '6')

    print('::: Problem 7 :::')
    seven_ret = mutation_medication_impact(data, attempt)
    util_5353.assert_dict(seven_ret, '7')
    util_5353.assert_int_eq(len(MUTATIONS), len(seven_ret), '7')
    for k,v in seven_ret.items():
      util_5353.assert_str(k, '7', valid_values=MUTATIONS)
      util_5353.assert_int_eq(len(MEDICATIONS), len(v), '7')
      for k2,v2 in seven_ret[k].items():
        util_5353.assert_str(k2, '7', valid_values=MEDICATIONS)
        util_5353.assert_float(v2, '7')
        util_5353.assert_float_range((-10.0, 10.0), v2, '7')

    print('::: Problem 8 :::')
    eight_ret = medication_impact(data, attempt)
    util_5353.assert_dict(eight_ret, '8')
    util_5353.assert_int_eq(len(MEDICATIONS), len(eight_ret), '8')
    for k,v in eight_ret.items():
      util_5353.assert_str(k, '8', valid_values=MEDICATIONS)
      util_5353.assert_float(v, '8')
      util_5353.assert_float_range((-20.0, 10.0), v, '8')

  print('~~~ All Tests Pass ~~~')



