# Bruce Willis
def expert_1(time_step):
	return 1

# glass is half empty
def expert_2(time_step):
	return -1

# odd-even rule 
def expert_3(time_step):
	if not time_step%2:
		return 1 # win if even
	else:
		return -1
