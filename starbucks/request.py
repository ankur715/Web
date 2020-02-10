import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'total_population':3641.35, 'median_age':34.88, 'median_hh_income':50491.48, 'median_rent':1032.15,
       'median_home_value':228266.33, 'percent_workers':0.48, 'percent_leave_7_9':0.45,
       'perc_hs_dipl':0.15, 'perc_bach_deg':0.14, 'perc_masters_deg':0.07,
       'perc_walk_to_work':0.06, 'perc_car_to_work':0.53, 'perc_pub_tran':0.25,
       'perc_bicycle_to_work':0.02, 'perc_work_from_home':0.04, 'time_to_work':34.13,
       'tran_mean':701.66})

print(r.json())

