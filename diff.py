import pandas as pd


def diff(modifiateData, initialData = None):
	#compare modifiateData in fonction of initialData
	#default initialData is ground_truth.csv

	if initialData == None:
		data1 = pd.read_csv("ground_truth.csv", sep=",")
	else:
		data1 = pd.read_csv(initialData, sep=",")
	data2 = pd.read_csv(modifiateData, sep=",")


	data1Columns = list(data1.columns)
	data1Rows = data1.shape[0]
	data2Columns = list(data2.columns)
	data2Rows = data2.shape[0]

	data1Score = []
	data2Score = []


	print("\n ----- STEP 1/2 ----- ")
	test = statOnUserID(data2, data2Columns, data2Rows, data2Score)

	print(" ----- STEP 2/2 ----- ")
	if test:
		statOnUserID(data1, data1Columns, data1Rows, data1Score)
	else:
		data1Score.pop()
		data1Score.append([])
		print(" -----> NO USERID")

	evaluate(data1Score, data2Score)


	

def statOnUserID(data, dataColumns, dataRows, dataScore):

	for columns in dataColumns:
		if columns == "id_user":
			nbRows = dataRows
			userIDFrequency = [[data["id_user"][0],0]]


			for i in range(nbRows):

				currentUserID = data["id_user"][i]
				print("-->",(int((i/nbRows)*10000)/100),"%",end='\r')
				isInUserIDFrequency = False
				for j in range(len(userIDFrequency)):
					if currentUserID == userIDFrequency[j][0]:
						userIDFrequency[j][1] += 1
						isInUserIDFrequency = True

				if not(isInUserIDFrequency):
					userIDFrequency.append([currentUserID,1])

			l = sorted(userIDFrequency, key = lambda x: int(x[1]))

			Q1 = int((0.25*len(userIDFrequency))+1)
			Q2 = int((0.5*len(userIDFrequency))+1)
			Q3 = int((0.75*len(userIDFrequency))+1)
			avgUserIDFreq = 0

			for k in range(len(userIDFrequency)):
				avgUserIDFreq += userIDFrequency[k][1]
			avgUserIDFreq = avgUserIDFreq/(len(userIDFrequency))
			userIDQ1 = l[Q1][1]
			userIDQ2 = l[Q2][1]
			userIDQ3 = l[Q3][1]

			res = [avgUserIDFreq, userIDQ1, userIDQ2, userIDQ3]

			dataScore.append(res)
			print(" ------> DONE")
			return 1



		else:
			print(" ----> NO USERID")
			dataScore.append([])
			return 0








def evaluate(initialDataScore, modifiateDataScore):


	aScore = initialDataScore
	bScore = modifiateDataScore
	totalScore = 0
	resFile = open("resDiff.txt",'w')

	resFile.write(" ---------- Differences between the 2 files ---------- \n")
	resFile.write("\n\n")
	resFile.write(" ---> Section UserID\n")
	resFile.write("\n")

	if len(aScore[0]) == 0:
		resFile.write(" NO USERID ")
	else :
		for i in range(len(aScore[0])):
			totalScore += (aScore[0][i]/bScore[0][i])
		totalScore = totalScore/len(aScore[0])

		resFile.write("Diff on avgUserIDFreq = " + str(round(((aScore[0][0]/bScore[0][0])*100),2)) + '\n')
		resFile.write("Diff on Q1 userID = " + str(round(((aScore[0][1]/bScore[0][1])*100),2)) + '\n')
		resFile.write("Diff on Q2 userID = " + str(round(((aScore[0][2]/bScore[0][2])*100),2)) + '\n')
		resFile.write("Diff on Q3 userID = " + str(round(((aScore[0][3]/bScore[0][3])*100),2)) + '\n')
		resFile.write("\n")


	resFile.write("\n")
	resFile.write(" ---------- Result diff ---------- \n")
	resFile.write("\n")
	resFile.write("           --> " + str(round(totalScore*100,2)) + " <-- \n")

	print(" -- ALL STEPS DONE -- ")
	print(" -> Result in resDiff.txt \n")

	















def main():
	diff("/home/clement/Documents/4a/projetSecurite/MMMMM/CSV_RENDU/S_Cochennec_v3.csv")

if __name__ == "__main__":
    main()