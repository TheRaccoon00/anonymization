def convert_list_to_CSV(path,gt_list):
    back_file = open(path,"w")
    for gt_line in gt_list:
        for index,element in enumerate(gt_line):
            if index == 0 and element == "DEL":
                back_file.write("DEL\n")
                break
            elif index != 5:
                back_file.write(str(element) + ",")
            else:
                back_file.write(element)

def delete_alone_list_items(gt_list,alone_file,modul_nb_trans_month):
    for alone_line in alone_file:
        for index_gt,gt_line in enumerate(gt_list):
            item_alone = alone_line[3]
            item_gt = gt_line[3]

            date_alone = alone_line[1]
            date_gt = gt_line[1]

            if gt_line[0] == alone_line[0] and item_alone == item_gt and date_alone == date_gt:
                gt_list.pop(index_gt)
                modul_nb_trans_month[date_alone[:7]] -= 1
                break
    print("#############")
    print(modul_nb_trans_month)
    print(len(gt_list))
    return gt_list
