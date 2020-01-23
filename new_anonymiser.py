from new_functions import *

start_file = "ground_truth.csv"
stages = [start_file,"out_new/stage1.csv","out_new/stage2.csv","out_new/stage3.csv","out_new/stage4.csv","out_new/stage5.csv","out_new/stage6.csv","out_new/stage7.csv"]
alones = [".alone_items_new"]

construction = 0
generate_mode = 0
distribute_mode = 0

# START MAIN
if construction == 1:
    pseudo(stages[0])
    del_hours(stages[1])
    modifyDateV2(stages[2])

if generate_mode == 1:
    generate_alone_items(stages[3])

if distribute_mode == 1:
    #distrib_items_v1(stages[3],alones[0])
    distrib_items_v2(stages[3],alones[0])
