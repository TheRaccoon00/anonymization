    #OLD MAIN
"""
pseudo('ground_truth.csv')
modificateDateV2('out/stage1.csv')
del_hours('out/stage2.csv')
generate_alone_items('out/stage3.csv')
#for nb_file in range(0,37):
#    filterV2('out/stage42.csv',".alone_items2/out"+str(nb_file)+".csv")
distribute_with_date("out/stage3.csv")
generate_alone_users('out/stage53.csv')
for nb_file in range(0,37):
    filterV2('out/stage6.csv',".alone_users/out"+str(nb_file)+".csv")
distribute_with_date('out/stage53.csv')
"""

def modificateDate(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    listDate = []
    for i in range(size):
        listDate.append(df["date"][i])

    dfn = np.asarray(df)

    for i in range(size):
        dfn[i][1] = listDate[i][:4] + "/" + listDate[i][5:7] + "/" + "01" # 1 is for date

    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out/stage2.csv', encoding='utf-8',index=False)


def filter(filename1,filename2):
    '''Reads the second file and deletes all the lines it contains in the first file'''
    file1 = open(filename1,"r")
    file2 = open(filename2,"r")
    listfile1 = []
    listfile1 = file1.readlines()
    listfile2 = []
    listfile2 = file2.readlines()
    for i in range(len(listfile1)):
        listfile2.append(0)
    for to_delete in listfile2:
        for original in listfile1:
            if to_delete == original:
                listfile1.remove(original)
    with open(filename1, 'w') as filteredfile:
        filteredfile.writelines("%s" % tuple for tuple in listfile1)

    file1.close()
    file2.close()

def filterV2(filename1,filename2):
    '''Reads the second file and deletes all the lines it contains in the first file'''
    file1 = open(filename1,"r")
    file2 = open(filename2,"r")
    listfile1 = []
    listfile1 = file1.readlines()
    listfile2 = []
    listfile2 = file2.readlines()
    for original in listfile1:
        if original in listfile2:
            listfile1.remove(original)
    with open(filename1, 'w') as filteredfile:
        filteredfile.writelines("%s" % tuple for tuple in listfile1)

    file1.close()
    file2.close()


def del_hours(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    for col in df.columns:
        df['hours'].values[:] = 'DEL'
    df.to_csv('out/stage3.csv',encoding='utf-8',index=False)

def generate_alone_items(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    date_dict = {}
    final_dict= {}
    dfn = np.asarray(df)
    for i in range(0,size):
        if df["date"][i] not in date_dict:
            date_dict[str(df["date"][i])] = []
        date_dict[str(df["date"][i])].append(dfn[i])

    save_filename = ".alone_items2/out"
    nb_file = 0

    for date in date_dict:
        out_file = open(save_filename + str(nb_file) +".csv","w")
        for i,id_item in enumerate(date_dict[str(date)]):
            compare = [date_dict[str(date)][i][3] for i in range(0, len(date_dict[str(date)]))]
            if compare.count(id_item[3]) <= 3:
                if date not in final_dict:
                    final_dict[str(date)] = []
                final_dict[str(date)].append(id_item)
                for index_item, item in enumerate(id_item):
                    if index_item != 5:
                        out_file.write(str(item) + ",")
                    else:
                        out_file.write(str(item) + "\n")

        out_file.close()
        nb_file += 1
        print(str(date) + ": " + str(len(final_dict[str(date)])))
    """
    files = list()
    files_list = list()

    alone_items = list()

    for nb_file in range(0,13):
        files.append(open(".alone_items/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)

    for file in files_list:
        for nuplet in file:
            compare = []
            for f in files_list:
                for line_compare in f:
                    compare.append(line_compare.split(",")[3])
            if compare.count(str(nuplet.split(",")[3])) <= 1:
                alone_items.append(nuplet)

    print(len(alone_items))
    """
    print("end")

def distribute_with_date(filename):

    file = open(filename,"r")
    file_list = []
    distribute_list = []
    distribute_list_str = []

    for line in file:
        file_list.append(line.split(","))

    files = list()
    files_list = list()

    for nb_file in range(0,37):
        files.append(open(".alone_users/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)

    count = 0
    found = 0

    for file in files_list:
        print("file")
        for indexfile, nuplet in enumerate(file):
            found = 0
            for index_modif,line_compare in enumerate(file_list):
                if nuplet.split(",") == line_compare:
                #if nuplet.split(",")[3] in nup.split(","):
                    list_modif = file_list[index_modif]
                    file_list[index_modif] = ["DEL"]
                    """
                    for index,element in enumerate(list_modif):
                        if index == 1:
                            file_list[index_modif].append("DEL")
                        else:
                            file_list[index_modif].append(element)
                    """
                    break

                    """
                    new_nuplet = nuplet.split(",")
                    nuplet_str = ""
                    for index,element in enumerate(new_nuplet):
                        if index == 1:
                            element = file_compare[0].split(",")[1]
                        if index != 5:
                            nuplet_str = nuplet_str + element + ","
                        else:
                            nuplet_str = nuplet_str + element + "\n"
                    #print(nuplet_str.strip())
                    count+=1
                    distribute_list_str.append(nuplet_str.strip())
                    """

    print("completing")

    """
    ##COMPLETE WITH FAKE LINES
    for line in distribute_list_str:
        distribute_list.append(line.split(","))

    line_number = 0
    max_number = len(distribute_list)
    while len(file_list) < 307055:
        file_list.append(distribute_list[line_number % max_number])
        line_number+=1
    """

    file_out = open("out/stage63.csv","w")
    for line in file_list:
        for index,element in enumerate(line):
            if index != 5 and len(line) > 1:
                file_out.write(element + ",")
            else:
                file_out.write(element.strip() + "\n")

    file_out.close()

def generate_alone_users(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    date_dict = {}
    final_dict= {}
    dfn = np.asarray(df)
    for i in range(0,size):
        if df["date"][i] not in date_dict:
            date_dict[str(df["date"][i])] = []
        date_dict[str(df["date"][i])].append(dfn[i])

    save_filename = ".alone_users/out"
    nb_file = 0

    for date in date_dict:
        out_file = open(save_filename + str(nb_file) +".csv","w")
        for i,id_item in enumerate(date_dict[str(date)]):
            compare = [date_dict[str(date)][i][0] for i in range(0, len(date_dict[str(date)]))]
            if compare.count(id_item[0]) <= 1:
                if date not in final_dict:
                    final_dict[str(date)] = []
                final_dict[str(date)].append(id_item)
                for index_item, item in enumerate(id_item):
                    if index_item != 5:
                        out_file.write(str(item) + ",")
                    else:
                        out_file.write(str(item) + "\n")

        out_file.close()
        nb_file += 1
        print(str(date) + ": " + str(len(final_dict[str(date)])))

    files = list()
    files_list = list()

    alone_items = list()

    for nb_file in range(0,13):
        files.append(open(".alone_users/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)

    for file in files_list:
        for nuplet in file:
            compare = []
            for f in files_list:
                for line_compare in f:
                    compare.append(line_compare.split(",")[0])
            if compare.count(str(nuplet.split(",")[0])) <= 1:
                alone_items.append(nuplet)

    print(len(alone_items))

def distribute_with_date_item(filename):

    file = open(filename,"r")
    file_list = []
    distribute_list = []
    distribute_list_str = []

    for line in file:
        file_list.append(line.split(","))
    print(len(file_list))

    files = list()
    files_list = list()

    for nb_file in range(0,37):
        files.append(open(".alone_users/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)
    count = 0
    found = 0
    for file in files_list:
        for indexfile, nuplet in enumerate(file):
            found = 0
            for file_compare in files_list:
                for nup in file_compare:
                    if nuplet.split(",")[0] in nup.split(","):
                        file.pop(indexfile)
                        new_nuplet = nuplet.split(",")
                        nuplet_str = ""
                        for index,element in enumerate(new_nuplet):
                            if index == 1:
                                element = file_compare[0].split(",")[1]
                            if index != 5:
                                nuplet_str = nuplet_str + element + ","
                            else:
                                nuplet_str = nuplet_str + element + "\n"
                        count+=1
                        distribute_list_str.append(nuplet_str.strip())
                        found = 1
                        break
                    if found == 1:
                        break

    ##COMPLETE WITH FAKE LINES
    for line in distribute_list_str:
        distribute_list.append(line.split(","))
    print(distribute_list)

    line_number = 0
    max_number = len(distribute_list)
    while len(file_list) < 307055:
        file_list.append(distribute_list[line_number % max_number])
        line_number+=1

    file_out = open("out/stage7.csv","w")
    for line in file_list:
        for index,element in enumerate(line):
            if index != 5:
                file_out.write(element + ",")
            else:
                file_out.write(element.strip() + "\n")

    file_out.close()

def id_user_id_item(filename):
    file = open(filename,"r")
    file_list = []
    repetitions = []

    for line in file:
        file_list.append(line.split(","))

    id_user_dict = {}
    final_dict= {}

    for line in file_list:
        if line[0] not in id_user_dict:
            id_user_dict[str(line[0])] = []
        id_user_dict[str(line[0])].append(line)

    for key in id_user_dict:
        repetitions.append((len(id_user_dict[key]),key))
    repetitions.sort()

    for rep in repetitions:
        print(rep)
    print(len(repetitions))
