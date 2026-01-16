import csv
import sys

reader = csv.reader(sys.stdin)
writer = csv.writer(sys.stdout)

for row in reader:
    writer.writerow(row[1:])
