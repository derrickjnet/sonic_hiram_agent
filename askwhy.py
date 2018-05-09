from graphdb import GraphDB
import numpy as np
import pandas as pd
gb = GraphDB('graph.db')

i = 0
n = 3
while i < 20:
    print('position',i)
    if i >= 0:
        da = pd.DataFrame(gb(i).has_action(list)).drop_duplicates()
        if not da.empty:
            print('top n actions by reward', da.nlargest(n,'curr_reward')) #Top 3 Actions
        dc = pd.DataFrame(gb(i).is_before_chron(list)).drop_duplicates()
        if not dc.empty:
            print('top n actions ', dc.nlargest(n,'curr_reward')) #Best future

        unstuck = gb('unstuck').at_place_spatial(list)
        df = pd.DataFrame(unstuck).drop_duplicates()
        # print('unstuck',df)
        #dc.loc[df['a'] > 10, ['a','c']]
        #observation,potential actions, results, potential
        #how many times has this move worked, how many times has this moved killed/done,
        #how does time play into this? min-max per action, std, sum, count, median. quantile, mean, var
        #rank, cumsum, cummax cummin


    i += 1


    # gb.store_relation(int(prev_loc), 'has_action', {'curr_action': curr_action, 'curr_reward': acts1})
    # gb.store_relation(int(prev_loc), 'is_before_chron',
    #                   {'curr_loc': self.curr_loc, 'curr_action': curr_action, 'curr_reward': acts1})
    # gb.store_relation('stuck', 'at_place_spatial',
    #                   {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
    #                    'curr_loc': self.curr_loc})
    # gb.store_relation('unstuck', 'at_place_spatial',
    #                   {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
    #                    'curr_loc': self.curr_loc})
    # gb.store_relation('act_of_god', 'at_place_spatial',
    #                   {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1})
