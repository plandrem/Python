import os
import csv

path = "C:\\\emaildump\\"

addresses = []

filt = lambda s: s.find('.eml') != -1

for f in filter(filt,os.listdir(path)):

	print f
	
	with open(os.path.join(path,f), 'r') as inF:
		for line in inF:
			if line.startswith('From:'):
				
				try:
					a = line.split('<')[1].split('>')[0]
					addresses.append(a)
				except:
					a = line.lstrip('From: ')
					a = a.rstrip()
					addresses.append(a)
				break	

addresses = set(addresses)

print addresses

with open('emaillist.txt','w') as outF:
	writer = csv.writer(outF)
	for a in addresses:
		writer.writerow([a])
