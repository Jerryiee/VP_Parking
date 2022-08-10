from influxdb_client import InfluxDBClient
from time import sleep


influx_db_url = "http://158.193.238.118:8086"
influx_db_token = "48XCBqMXJeSqW8yRe857wrtqu-X0MGuyF3GQWJc0TfpvIJrnqAlcPY81XZcHJSIDdk9bWGfYmpc6uwOEvKTsng=="
influx_db_org = "uvpuniza"


influx_db = InfluxDBClient(url=influx_db_url, token=influx_db_token, org=influx_db_org)
query_api = influx_db.query_api()

query = 'from(bucket: "clevernet") \
|> range(start:-15s) \
|> filter(fn: (r) => r["_measurement"] == "traffic") // car / van / truck: \
|> filter(fn: (r) => r["_field"] == "len0_speed1_cnt12" or r["_field"] == "len1_speed1_cnt12" or r["_field"] == "len2_speed1_cnt12" \
|>    or r["_field"] == "len0_speed2_cnt12" or r["_field"] == "len1_speed2_cnt12" or r["_field"] == "len2_speed2_cnt12" \
|>    or r["_field"] == "len0_speed3_cnt12" or r["_field"] == "len1_speed3_cnt12" or r["_field"] == "len2_speed3_cnt12" \
|>    or r["_field"] == "len0_speed0_cnt12" or r["_field"] == "len1_speed0_cnt12" or r["_field"] == "len2_speed0_cnt12") \
|> filter(fn: (r) => r["deviceName"] == "dd-8112573d" or r["deviceName"] == "dd-8112578f")  // UNIZA.park.IN, UNIZA.park.OUT \
|> aggregateWindow(every: 5s, fn: mean, createEmpty: false) \
|> difference(keepFirst:false, nonNegative:true) \
|> yield(name: "mean")'

while True:
    result = influx_db.query_api().query_stream(org=influx_db_org, query=query)
    for record in result:
        print(f'Is {record.get_time()}')
    sleep(5) #test odozvy

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