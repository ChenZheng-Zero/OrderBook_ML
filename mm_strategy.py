import numpy as np
import pandas as pd
from utils import time_to_int, int_to_time
import pdb

def market_making(trd_filename, full_test_data, full_test_timestamp, full_test_labels, smart, max_holdings=100, unit=1, spread=0.01):
	# read-in the RMD trade data
	trades = pd.read_excel(trd_filename).sort_values('Time').reset_index()
	#pdb.set_trace()
	cash = 0
	position = 0
	buy = 0
	sell = np.inf
	buy_boolean = 0
	sell_boolean = 0
	src = ""

	i = 0
	trd_i = 0
	# TO-DO check quantity match
	while i < len(full_test_labels)-1:
		# move transaction to the right timestamp
		while trd_i < trades.shape[0]-1 and time_to_int(trades['Time'][trd_i]) < full_test_timestamp[i]:
			trd_i = trd_i+1
		
		# check transaction 
		# a)check trades;
		while trd_i < trades.shape[0]-1 and time_to_int(trades['Time'][trd_i]) == full_test_timestamp[i]:
			if trades['BUY_SELL_FLAG'][trd_i] == 0 and trades['PRICE'][trd_i] >= sell:
				#print("Sell at price {} with TRD at time {}. Current cash is {} and current holding is {}".format(sell, int_to_time(full_test_timestamp[i]), format(cash+sell, '.3f'), position-1))
				src = "TRD"
				sell_boolean = 1
			if trades['BUY_SELL_FLAG'][trd_i] == 1 and trades['PRICE'][trd_i] <= buy:
				#print("Buy at price {} with TRD at time {}. Current cash is {} and current holding is {}".format(buy, int_to_time(full_test_timestamp[i]), format(cash-buy, '.3f'), position+1))
				src = "TRD"
				buy_boolean = 1
			trd_i = trd_i+1
		# b)check orderbook 
		if full_test_data[i][0] <= buy:
			#print("Buy at price {} with OB at time {}. Current cash is {} and current holding is {}".format(buy, int_to_time(full_test_timestamp[i]), format(cash-buy, '.3f'), position+1))
			src = "OB"
			buy_boolean = 1
		if full_test_data[i][2] >= sell:
			#print("Sell at price {} with OB at time {}. Current cash is {} and current holding is {}".format(sell, int_to_time(full_test_timestamp[i]), format(cash+sell, '.3f'), position-1))
			src = "OB"
			sell_boolean = 1

		if buy_boolean == 1:
			position = position + 1
			cash = cash - buy
			print("Buy at price {} from {} at time {}. Current cash is {} and current holding is {}".format(buy, src, int_to_time(full_test_timestamp[i]), format(cash, '.3f'), position))
		if sell_boolean == 1:
			position = position - 1
			cash = cash + sell
			print("Sell at price {} from {} at time {}. Current cash is {} and current holding is {}".format(sell, src, int_to_time(full_test_timestamp[i]), format(cash, '.3f'), position))
		
		# update market making orders
		if round(full_test_data[i][0] - full_test_data[i][2], 3) >= 0.02 + spread:
			if smart:
				if full_test_labels[i] == 0 or full_test_labels[i] == -1:
					buy = full_test_data[i][2]+0.01
					sell = full_test_data[i][0]-0.01
				else:
					buy = full_test_data[i][2]+0.01
					sell = full_test_data[i][0]
				#else:
				#	buy = full_test_data[i][2]
				#	sell = full_test_data[i][0]-0.01
			else:
				buy = full_test_data[i][2] + 0.01
				sell = full_test_data[i][0] - 0.01
		else:
			if full_test_data[i][0] > full_test_data[i][2]:
				if smart:
					if full_test_labels[i] == 0 or full_test_labels[i] == -1:
						buy = full_test_data[i][2]
						sell = full_test_data[i][0]
					else:
						buy = full_test_data[i][2]
						sell = full_test_data[i][0]+0.01
					#else:
					#	buy = full_test_data[i][2]-0.01
					#	sell = full_test_data[i][0]
				else:
					buy = full_test_data[i][2]
					sell = full_test_data[i][0]
			else:
				print("Rare")
				buy = 0
				sell = np.inf
		#print("Market making with buy at price {} and sell at price {} at time {}, current quote is {} and {}.".format(buy, sell, int_to_time(full_test_timestamp[i]), full_test_data[i][2], full_test_data[i][0]))
		
		# re-initialize
		buy_boolean = 0
		sell_boolean = 0
		i = i+1
	
	if position > 0:
		cash = cash + position * full_test_data[i][2]
	else:
		cash = cash + position * full_test_data[i][0]
	print(cash)
	print(position)
	return cash, position