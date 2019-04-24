import os
import preprocess
import test
import record
import train

def main():
	print("Welcome to Speech2SQL system. Senior Capstone Project by Ashwin Mishra")
	print("=======================================================================")
	print("1 train the model")
	print("2 test on example audio files")
	print("3 record live")
	print("=======================================================================")
	option = input("Please select what you want to do!\n")
	if (option == "1"):
		print("=======================================================================")
		prep = preprocess.Preprocess()
		prep.run()
		# train_ = train.Train()
		# train_.run()
	elif (option == "2"):
		print("=======================================================================")
		test_ = test.Test()
		test_.run()
	elif (option == "3"):
		print("=======================================================================")
		record_ = record.Record()
		record_.run()
	else:
		print("Please select a valid option [1,2,3]!")

main()