from influxdb_client import Point, InfluxDBClient
import os
import pandas as pd
import numpy as np

influx_db_url = "http://158.193.238.118:8086"
influx_db_token = "48XCBqMXJeSqW8yRe857wrtqu-X0MGuyF3GQWJc0TfpvIJrnqAlcPY81XZcHJSIDdk9bWGfYmpc6uwOEvKTsng=="
influx_db_org = "uvpuniza"


influx_db = InfluxDBClient(url=influx_db_url, token=influx_db_token, org=influx_db_org, debug=False)
query_api = influx_db.query_api()

# Must contain PIVOT function!!!
query2 = '''
from(bucket: "clevernet")
    |> range(start: -8h)
    |> filter(fn: (r) => r._measurement == "traffic"
                    and (r.deviceName == "dd-8112573d" 
                        or r.deviceName == "dd-8112578f"
                        )
                    and (r._field == "len0_speed1_cnt12"
                        or r._field == "len1_speed1_cnt12"
                        or r._field == "len2_speed1_cnt12"
                        or r._field == "len3_speed1_cnt12"
                        )                        
                    )
    |> difference(keepFirst: false, nonNegative: true)
    |> drop(columns: ["_start", "_stop", "_measurement"])    
    |> pivot(rowKey: ["_time", "deviceName"], columnKey: ["_field"], valueColumn: "_value")
    |> rename(columns: {_time: "Time",  
            deviceName: "SensorName",
            len0_speed1_cnt12: "CarsLT40", 
            len1_speed1_cnt12: "VansLT40",
            len2_speed1_cnt12: "TrucksLT40"
        })
    |> yield(name: "after pivot")
'''
if not os.path.isdir("csv"): #create dic for save file
  os.makedirs("csv")

result = query_api.query_data_frame(query=query2)
result.head()
in_veh = result[result['SensorName'] == 'dd-8112573d'].copy() #IN&out filter
out_veh = result[result['SensorName'] == 'dd-8112578f'].copy()


in_veh.loc['total'] = in_veh.select_dtypes(np.number).sum() #sum
out_veh.loc['total'] = out_veh.select_dtypes(np.number).sum()

print(in_veh)
print(out_veh)

in_veh.to_csv('csv/datain.csv', index=False)
out_veh.to_csv('csv/dataout.csv', index=False)