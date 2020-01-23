from new_functions import *
from construction import *
from items import *
from id_users import *

start_file = "datas/ground_truth.csv"
stages = [start_file,"out_new/stage1.csv","out_new/stage2.csv","out_new/stage3.csv","out_new/stage4.csv","out_new/stage5.csv","out_new/stage6.csv","out_new/stage7.csv"]
alones = [".alone_items_new",".alone_users_new",".alone_alone"]

construction = 0
generate_item_mode = 0
distribute_item_mode = 0
generate_user_mode = 0
distribute_user_mode = 1

# START MAIN
if construction == 1:
    pseudo(stages[0])
    del_hours(stages[1])
    modifyDateV2(stages[2])

if generate_item_mode == 1:
    generate_alone_items(stages[3])

if distribute_item_mode == 1:
    #distrib_items_v1(stages[3],alones[0])
    distrib_items_v2(stages[3],alones[0])

if generate_user_mode == 1:
    generate_alone_users(stages[3])

if distribute_user_mode == 1:
    delete_alone_users(stages[6],alones[1])
