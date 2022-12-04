import csv

datas = []
dic = {'WWWW': 'FFFF', 'MSS': '2', 'TTL': '40', 'WS': '3', 'S': 1, 'N': 1, 'D': 0, 'T': 8, 'F': 'S', 'LEN': '3C'}
header = list(dic.keys())
datas.append(dic)

with open('test.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
    writer.writeheader()  # 写入列名
    writer.writerows(datas)  # 写入数据