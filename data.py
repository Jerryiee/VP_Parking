from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from IPython.display import display
from time import sleep


influx_db_url = "http://158.193.238.118:8086"
influx_db_token = "48XCBqMXJeSqW8yRe857wrtqu-X0MGuyF3GQWJc0TfpvIJrnqAlcPY81XZcHJSIDdk9bWGfYmpc6uwOEvKTsng=="
influx_db_org = "uvpuniza"


influx_db = InfluxDBClient(url=influx_db_url, token=influx_db_token, org=influx_db_org, debug=False)
query_api = influx_db.query_api()

query = 'from(bucket: "clevernet") \
|> range(start:-15s) \
|> filter(fn: (r) => r["_measurement"] == "traffic")\
|> filter(fn: (r) => r["_field"] == "len0_speed1_cnt12" or r["_field"] == "len1_speed1_cnt12" or r["_field"] == "len2_speed1_cnt12") // car / van / truck\
|> filter(fn: (r) => r["deviceName"] == "dd-8112573d")\
|> aggregateWindow(every: 5m, fn: mean, createEmpty: false)\
|> difference(keepFirst:false, nonNegative:true)\
|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value") \
|> yield(name: "mean")' 


result = query_api.query_data_frame(org=influx_db_org, query=query)
while True:
    try:
        result = query_api.query_data_frame(org=influx_db_org, query=query)
        new_result= result.iloc[:, [4,5,6,8]]
        print(new_result)
    except:
        print("An exception occurred, trying again..")
        

#df = DataFrame(resoverall.fetchall())
#df.columns = resoverall.keys()

'''
result = influx_db.query_api().query(org=influx_db_org, query=query)

resultset = []
for table in result:
    for record in table.records:
            resultset.append((record.get_value(), record.get_field()))
            
    print(resultset)



'''