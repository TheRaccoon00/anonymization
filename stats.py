import pandas as pd
import time
from operator import itemgetter

def displayStatusBar(i, nbRows, l):
	treated = int((i/nbRows)*100)

	if i == 0:
		spinning_slatch = "|"

	if (treated not in l) and (treated%2 == 0):

		if (treated/2)%4 == 0:
			spinning_slatch = "| "
		elif (treated/2)%4 == 1:
			spinning_slatch = "/ "
		elif (treated/2)%4 == 2:
			spinning_slatch = "-"
		elif (treated/2)%4 == 3:
			spinning_slatch = "\\ "

		display = "#"*len(l) + "-"*(49-len(l))
		l.append(treated)
		print("[",display,"]","  ",spinning_slatch,end="\r")

	if i >= nbRows - 1:
		print("----------------------------- Result -----------------------------")



def statOnUserID(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]

	minUserID = data["id_user"][0]
	maxUserID = data["id_user"][0]
	mostFrequentUserID = data["id_user"][0]
	nbMostFrequentUserID = 1
	lessFrequentUserID = data["id_user"][0]
	nbLessFrequentUserID = 1
	avgUserID = data["id_user"][0]
	userIDFrequency = [[data["id_user"][0],0]]


	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)


		currentUserID = data["id_user"][i]

		if currentUserID < minUserID:
			minUserID = currentUserID

		if currentUserID > maxUserID:
			maxUserID = currentUserID


		isInUserIDFrequency = False
		for j in range(len(userIDFrequency)):
			if currentUserID == userIDFrequency[j][0]:
				userIDFrequency[j][1] += 1
				isInUserIDFrequency = True

		if not(isInUserIDFrequency):
				userIDFrequency.append([currentUserID,1])

		avgUserID += currentUserID

	for k in range(len(userIDFrequency)):
		currentUserID = userIDFrequency[k]

		if currentUserID[1] > nbMostFrequentUserID:
			mostFrequentUserID = currentUserID[0]
			nbMostFrequentUserID = currentUserID[1]

		if currentUserID[1] < nbLessFrequentUserID:
			lessFrequentUserID = currentUserID[0]
			nbLessFrequentUserID = currentUserID[1]

	avgUserID = avgUserID/nbRows


	print("minUserID :",minUserID)
	print("maxUserID :",maxUserID)
	print("mostFrequentUserID :",mostFrequentUserID,"frequency :",nbMostFrequentUserID)
	print("lessFrequentUserID :",lessFrequentUserID,"frequency :",nbLessFrequentUserID)
	print("avgUserID :",avgUserID)





def statOnDate(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]


	avgDate = [int(data["date"][0][:4]),int(data["date"][0][5:7]),int(data["date"][0][8:])]
	minDate = data["date"][0]
	maxDate = data["date"][0]
	frequencyByYear = [[data["date"][0][:4],0]]
	frequencyByMonth = [[data["date"][0][5:7],0]]
	frequencyByDay = [[data["date"][0][8:],0]]

	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)


		currentDate = data["date"][i]

		avgDate[0] += int(currentDate[:4])
		avgDate[1] += int(currentDate[5:7])
		avgDate[2] += int(currentDate[8:])

		if time.strptime(currentDate, "%Y/%m/%d") > time.strptime(maxDate, "%Y/%m/%d"):
			maxDate = currentDate

		if time.strptime(currentDate, "%Y/%m/%d") < time.strptime(minDate, "%Y/%m/%d"):
			minDate = currentDate

		isInFrequencyByYear = False
		for j in range(len(frequencyByYear)):
			if frequencyByYear[j][0] == currentDate[:4]:
				frequencyByYear[j][1] += 1
				isInFrequencyByYear = True

		if not(isInFrequencyByYear):
			frequencyByYear.append([currentDate[:4],1])

		isInFrequencyByMonth = False
		for j in range(len(frequencyByMonth)):
			if frequencyByMonth[j][0] == currentDate[5:7]:
				frequencyByMonth[j][1] += 1
				isInFrequencyByMonth = True

		if not(isInFrequencyByMonth):
			frequencyByMonth.append([currentDate[5:7],1])

		isInFrequencyByDay = False
		for j in range(len(frequencyByDay)):
			if frequencyByDay[j][0] == currentDate[8:]:
				frequencyByDay[j][1] += 1
				isInFrequencyByDay = True

		if not(isInFrequencyByDay):
			frequencyByDay.append([currentDate[8:],1])



	avgDate[0] = avgDate[0]/nbRows
	avgDate[1] = avgDate[1]/nbRows
	avgDate[2] = avgDate[2]/nbRows


	print("avgDate :",avgDate[0],"/",avgDate[1],"/",avgDate[2])
	print("minDate :",minDate)
	print("maxdate :",maxDate)
	print("frequencyByYear :",frequencyByYear)
	print("frequencyByMonth :",frequencyByMonth)
	print("frequencyByDay :",frequencyByDay)




def statOnHours(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]

	avgTime = [int(data["hours"][0][:2]),int(data["hours"][0][3:])]
	frequencyByHour = [[data["hours"][0],0]]

	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)

		currentTime = data["hours"][i]

		avgTime[0] += int(currentTime[:2])
		avgTime[1] += int(currentTime[3:])


		isInFrequencyByHour = False
		for j in range(len(frequencyByHour)):
			if (frequencyByHour[j][0][:2] == currentTime[:2]) and (frequencyByHour[j][0][3:] == currentTime[3:]):
				frequencyByHour[j][1] += 1
				isInFrequencyByHour = True

		if not(isInFrequencyByHour):
			frequencyByHour.append([currentTime,1])

	avgTime[0] = avgTime[0]/nbRows
	avgTime[1] = avgTime[1]/nbRows

	print("avgTime :",avgTime[0],":",avgTime[1])
	print("frequencyByHour :",frequencyByHour)







def statOnItemID(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]

	frequencyItemID = [[data["id_item"][0],0]]


	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)

		currentItemID = data["id_item"][i]


		isInFrequencyItemID = False
		for j in range(len(frequencyItemID)):
			if frequencyItemID[j][0] == currentItemID:
				frequencyItemID[j][1] += 1
				isInFrequencyItemID = True

		if not(isInFrequencyItemID):
			frequencyItemID.append([currentItemID,1])

	sorted(frequencyItemID,key=itemgetter(1))
	print("frequencyItemID : ",frequencyItemID)





def statOnPrice(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]

	minPrice = data["price"][0]
	maxPrice = data["price"][0]
	avgPrice = data["price"][0]
	frequencyPrice = [[data["price"][0],0]]


	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)

		currentPrice = data["price"][i]

		if currentPrice > maxPrice:
			maxPrice = currentPrice

		if currentPrice < minPrice:
			minPrice = currentPrice

		avgPrice += currentPrice

		isInFrequencyPrice = False
		for j in range(len(frequencyPrice)):
			if frequencyPrice[j][0] == currentPrice:
				frequencyPrice[j][1] += 1
				isInFrequencyPrice = True

		if not(isInFrequencyPrice):
			frequencyPrice.append([currentPrice,1])

	avgPrice = avgPrice/nbRows

	print("minPrice : ",minPrice)
	print("maxPrice : ",maxPrice)
	print("avgPrice : ",avgPrice)
	print("frequencyPrice : ",frequencyPrice)



def statOnQty(data, l=None):

	if l == None:
		l = []

	nbRows = data.shape[0]

	minQty = data["qty"][0]
	maxQty = data["qty"][0]
	avgQty = data["qty"][0]
	frequencyQty = [[data["qty"][0],0]]


	for i in range(nbRows):

		displayStatusBar(i, nbRows, l)

		currentQty = data["qty"][i]

		if currentQty > maxQty:
			maxQty = currentQty

		if currentQty < minQty:
			minQty = currentQty

		avgQty += currentQty

		isInFrequencyQty = False
		for j in range(len(frequencyQty)):
			if frequencyQty[j][0] == currentQty:
				frequencyQty[j][1] += 1
				isInFrequencyQty = True

		if not(isInFrequencyQty):
			frequencyQty.append([currentQty,1])

	avgQty = avgQty/nbRows

	print("minQty : ",minQty)
	print("maxQty : ",maxQty)
	print("avgQty : ",avgQty)
	print("frequencyQty : ",frequencyQty)




def main():
	data = pd.read_csv("ground_truth.csv", sep=",")

	data = pd.read_csv("/home/theoguidoux/INSA/ws/projetsecu4a/docs/CSV_RENDU/S_Godille_Table_1.csv", sep=",")

	months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
	days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
	l = []


	statOnUserID(data)
	statOnDate(data)
	statOnHours(data)
	statOnItemID(data)
	statOnPrice(data)
	statOnQty(data)

	# Fonctions are here !
	"""
	statOnUserID(data)
	statOnDate(data)
	statOnHours(data)
	statOnItemID(data)
	statOnPrice(data)
	statOnQty(data)
	"""

if __name__ == "__main__":
    main()
