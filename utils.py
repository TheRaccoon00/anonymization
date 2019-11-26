import random
import time
import os
import copy

def pimpedRandom():
	n = random.randrange(0,1000000000000000000)

	m = time.time() - 1569650000

	n = n * m

	os.system("cat /proc/meminfo | grep MemFree > /tmp/randomGen.txt");
	f = open("/tmp/randomGen.txt","r")
	res = f.read()
	l = [int(s) for s in res.split() if s.isdigit()]

	for i in range(len(l)):
		if l[i]!=0:
			n = n * l[i]

	os.system("cat /proc/uptime > /tmp/randomGen.txt");
	f = open("/tmp/randomGen.txt","r")
	res = f.read()
	l = [int(s) for s in res.split() if s.isdigit()]

	for i in range(len(l)):
		if l[i]!=0:
			n = n - l[i]

	os.system("cat /proc/buddyinfo > /tmp/randomGen.txt");
	f = open("/tmp/randomGen.txt","r")
	res = f.read()
	l = [int(s) for s in res.split() if s.isdigit()]

	for i in range(len(l)):
		if l[i]!=0:
			n = n * l[i]


	os.system("cat /proc/partitions > /tmp/randomGen.txt");
	f = open("/tmp/randomGen.txt","r")
	res = f.read()
	l = [int(s) for s in res.split() if s.isdigit()]
	n = n + l[1]

	n = int(n %(l[2])) * l[3] * l[4] * l[5] + l[6]
	n = n - random.randrange(0,1000000)

	return n

def sortedByColumn1(data,c):

	resData = copy.deepcopy(data)
	columnList = []

	for i in range(len(data)):
		columnList.append(data[i][c])
	columnList.sort()
	for i in range(len(columnList)):
		for j in range(len(data)):
			if data[j][c] == columnList[i]:
				resData[i] = data[j]
				break
	return resData

def sortedByColumn2(data,c):

	resData = copy.deepcopy(data)
	columnList = []

	for i in range(len(data)):
		columnList.append(data[i][c])
	columnList.sort()
	print("START SORT")
	for i in range(len(columnList)):
		if i % 1000 == 0:
			print("i = ",i,end='\r')
		for j in range(len(data)):
			if data[j][c] == columnList[i]:
				resData[i] = data[j]
				break
	return resData
		




