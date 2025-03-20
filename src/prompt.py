

def prompt_tool_retriever(tool_name, tool_description):
    if tool_description != "":
        return '''tool name: {}, tool description: {}'''.format(tool_name, tool_description)
    else:
        return '''tool name: {}'''.format(tool_name)

prompt_gsm8k_xl_infer = '''Answer the following questions step by step

Question: Mark has 3 tanks for pregnant fish.  Each tank has 4 pregnant fish and each fish gives birth to 20 young.  How many young fish does he have at the end?
Answer: He has 4*3=12 pregnant fish They give birth to 12*20=240 fish #### 240

Question: The math questions in a contest are divided into three rounds: easy, average, and hard. There are corresponding points given for each round. That is 2, 3, and 5 points for every correct answer in the easy, average, and hard rounds, respectively. Suppose Kim got 6 correct answers in the easy; 2 correct answers in the average; and 4 correct answers in the difficult round, what are her total points in the contest?
Answer: Kim got 6 points/round x 2 round = 12 points in the easy round. She got 2 points/round x 3 rounds = 6 points in the average round. She got 4 points/round x 5 rounds = 20 points in the difficult round. So her total points is 12 points + 6 points + 20 points = 38 points. #### 38

Question: A clothing store sells 20 shirts and 10 pairs of jeans. A shirt costs $10 each and a pair of jeans costs twice as much. How much will the clothing store earn if all shirts and jeans are sold?
Answer: Twenty shirts amount to $10 x 20 = $200. The cost of each pair of jeans is $10 x 2 = $20. So 10 pairs of jeans amount to $20 x 10 = $200. Therefore, the store will earn $200 + $200 = $400 if all shirts and jeans are sold. #### 400

Question: Arnold's collagen powder has 18 grams of protein for every 2 scoops.  His protein powder has 21 grams of protein per scoop.  And his steak has 56 grams of protein.   If he has 1 scoop of collagen powder, 1 scoop of protein powder and his steak, how many grams of protein will he consume?
Answer: 2 scoops of collagen powder have 18 grams of protein and he only has 1 scoop so he consumes 18/2 = 9 grams of protein He has 9 grams collagen powder, 21 grams of protein powder and 56 grams in his steak for a total of 9+21+56 = 86 grams of protein #### 86

Question: {}
Answer:{}'''

prompt_gsm8k_xl_tool_mode = '''{} Let's think step by step.{}'''


prompt_funcqa_infer_oh = '''Q: If Amy's income increases by 4% annually, how many times will it multiply in 11 years?
A: In 11 years, Amy's income will increase by 1.04^11=1.54 times. So, the answer is 1.54.

Q: If a store sells 147 bananas today and 354 more bananas tomorrow, how many bananas does the store sell in total?
A: The store sells 147 bananas today and 354 more bananas tomorrow, so the total number of bananas sold is 147+354=501. So, the answer is 501.

Q: A man had 789.4 dollars in his wallet. He spent 11.99 dollars on a movie ticket. How much money does he have left now?
A: The man had 789.4 dollars in his wallet and spent 11.99 dollars on a movie ticket, so he has 789.4-11.99=777.41 dollars left. So, the answer is 777.41 dollars.

Q: If a cake weighs 3.77 pounds and is divided into 13 equal pieces, how much does each piece weight?
A: Each piece of the cake weighs 3.77/13=0.29 pounds. So, the answer is 0.29 pounds.

Q: {}
A:{}'''

prompt_funcqa_infer_mh = '''Answer the following questions step by step:

Question: A coin is tossed 8 times, what is the probability of getting exactly 7 heads ?
Answer: The total number of possible outcomes to toss a coin 8 times is 2^8=256. The number of ways of getting exactly 7 heads is 8. The probability of getting exactly 7 heads is 8/256=0.03125. #### 0.03125

Question: If paint costs $3.2 per quart, and a quart covers 12 square feet, how much will it cost to paint the outside of a cube 10 feet on each edge?
Answer: The total surface area of the 10 ft cube is 6*10^2=6*100=600 square feet. The number of quarts needed is 600/12=50. The cost is 50*3.2=160. #### 160

Question: log(x)=2, log(y)=0.1, what is the value of log(x-y) ?
Answer: log(x)=2, so x=10^2=100; log(y)=0.1, so y=10^0.1=1.26; x-y=100-1.26=98.74, so log(x-y)=log(98.74)=1.99. #### 1.99

Question: How many degrees does the hour hand travel when the clock goes 246 minutes?
Answer: The hour hand travels 360 degrees in 12 hours, so every hour it travels 360/12=30 degrees. 246 minutes is 246/60=4.1 hours. The hour hand travels 4.1*30=123 degrees. #### 123

Question: {}
Answer:{}'''

prompt_funcqa_tool_mode = '''Q: {}\nA: {}'''


prompt_kamel_infer = '''Question: {}\nAnswer: The answer is{}'''

prompt_kamel_tool_mode = '''Question: {}\nAnswer: The answer is{}'''


# prompt_SQ_tool_mode = '''Question: {}\nAnswer: The answer is{}'''
prompt_SQ_tool_mode = '''Question: {}\nAnswer: The answer is{}'''

# "Answer the following question with the operator <lcm>:\n\nQ: A factory produces two types of products. The first product is produced every 865 days, and the second product is produced every 460 days. If the factory starts producing both products at the same time, what is the least amount of time it will take for the factory to produce both products on the same day?\nA: The least amount of time it will take for the factory to produce both products on the same day is <lcm>(865,460)=79580.\n\nQ: A music teacher wants to schedule her students' practice sessions on a rotation basis. One student needs to practice every 221 days, another student every 172 days, and the third student every 250 days. If the teacher wants the students to practice together on the same day, what is the least number of days before they can all practice together again?\nA: The least number of days before they can all practice together again is the least common multiple of 221, 172, and 250, which is <lcm>(221,172,250)=4751500.\n\nQ: A company has two machines that produce a product. Machine A produces one unit every 284 minutes, while Machine B produces one unit every 95 minutes. If both machines start at the same time, what is the least amount of time it will take for both machines to produce the same number of units?\nA: The least amount of time it will take for both machines to produce the same number of units is <lcm>(284,95)=26980.\n\nQ: A library has two bookshelves, one with books that are 643 pages long and the other with books that are 747 pages long. If both bookshelves need to be reorganized at the same time, what is the least amount of time it will take to reorganize both bookshelves?\nA: The least amount of time it will take to reorganize both bookshelves is <lcm>(643,747)=480321.\n\nQ: [QUESTION]\nA:"

# "Answer the following question with the operator <lcm>:\n\nQ: A factory produces two types of products. The first product is produced every 865 days, and the second product is produced every 460 days. If the factory starts producing both products at the same time, what is the least amount of time it will take for the factory to produce both products on the same day?\nA: The least amount of time it will take for the factory to produce both products on the same day is 865 days + 460 days = <lcm>(865,460)=79580.\n\nQ: A music teacher wants to schedule her students' practice sessions on a rotation basis. One student needs to practice every 221 days, another student every 172 days, and the third student every 250 days. If the teacher wants the students to practice together on the same day, what is the least number of days before they can all practice together again?\nA: <lcm>(221,172,250)=4751500.\n\nQ: [QUESTION]\nA:"

# "Answer the following question with the operator <lcm>:\n\nQ: A teacher wants to give three different assignments. Assignment A cycles every 38 days, and Assignment B cycles every 49 days, and Assignment C cycles every 28 days. If the teacher gives all three assignments on the same day, how many days will it take for the teacher to give all three assignments on the same day again?\nA: <lcm>(38,49,28)=3724.\n\nQ: [QUESTION]\nA:"  43

# "Answer the following question with the operator <lcm>:\n\nQ: A professor wants to give three different tasks. Task A cycles every 21 days, and task B cycles every 18 days, and tasks C cycles every 35 days. If the professor gives all three tasks on the same day, how many days will it take for the professor to give all three tasks on the same day again?\nA: <lcm>(21,18,35)=630.\n\nQ: [QUESTION]\nA:" 41.67