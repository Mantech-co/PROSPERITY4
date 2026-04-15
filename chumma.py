import csv

f = open('data.csv', 'r')
total = 0
count = 0
k = csv.reader(f)
for i in k:
    try:
        total += float(i[1])
        count += 1
    except:
        print(i)

print(total/count)